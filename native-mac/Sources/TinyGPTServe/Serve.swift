import Foundation
import Darwin
import MLX
import MLXRandom
import TinyGPTIO
import TinyGPTModel

// MARK: - lm-eval-harness adapter
//
// `tinygpt serve` exposes an OpenAI-compatible HTTP endpoint over a loaded
// tinygpt / HF model. This unlocks running the canonical `lm-evaluation-harness`
// against any tinygpt model — HellaSwag, MMLU-Pro, GSM8K, IFEval, GPQA-Diamond
// — by pointing the harness at `local-chat-completions` with our base_url.
//
// Wire-up:
//   POST /v1/chat/completions     — chat-style requests (messages: [...])
//   POST /v1/completions          — plain text completion (prompt: "...")
//   GET  /v1/models               — list "tinygpt" so clients can probe readiness
//
// Implementation notes:
//   - Uses POSIX sockets (Darwin) — zero new dependencies, works on macOS 14+.
//   - One thread per connection. Sample throughput dominates anyway; the
//     request rate from a single `lm-eval` driver is sequential.
//   - JSON parse/encode via Foundation. SSE streaming is supported when
//     the request body has `stream: true` — emits one event per token in
//     OpenAI's `chat.completion.chunk` format. lm-evaluation-harness
//     itself doesn't need streaming for loglikelihood / generate-until
//     tasks, but realtime/interactive clients do (the north-star
//     interaction model).
//   - All MLX work is serialised on a single dispatch queue (`inferenceQueue`)
//     because the model + KV cache are not thread-safe.
//
// TODO(merge): wire `case "serve": Serve.run(args: Array(args.dropFirst()))`
// into `TinyGPT.swift`'s subcommand dispatch. Left out per agent-coordination
// rules — another agent is touching that dispatch table.
//
// Tested via `Tests/TinyGPTServeTests/TinyGPTServeTests.swift` which boots a
// server on a random port, fires curl-equivalent NSURLSession requests, and
// asserts the JSON shape matches the OpenAI spec.
public enum Serve {
    public static func run(args: [String]) {
        var modelPath: String? = nil
        var host = "127.0.0.1"
        var port: UInt16 = 8080
        var maxContext: Int? = nil
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--port":
                guard i + 1 < args.count, let p = UInt16(args[i + 1]) else { exitUsage() }
                port = p; i += 2
            case "--host":
                guard i + 1 < args.count else { exitUsage() }
                host = args[i + 1]; i += 2
            case "--max-context":
                guard i + 1 < args.count, let n = Int(args[i + 1]) else { exitUsage() }
                maxContext = n; i += 2
            case "-h", "--help":
                exitUsage()
            default:
                if args[i].hasPrefix("-") {
                    fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
                }
                modelPath = args[i]; i += 1
            }
        }
        guard let modelPath = modelPath else {
            fputs("serve: missing <model path>\n", stderr); exitUsage()
        }

        do {
            let server = try Server.boot(modelPath: modelPath, host: host, port: port,
                                          maxContextOverride: maxContext)
            print("tinygpt serve — listening on http://\(host):\(server.port)")
            print("model: \(modelPath)  ·  ctx=\(server.maxContext)  ·  vocab=\(server.config.vocabSize)")
            // Block forever — the listener thread runs detached.
            dispatchMain()
        } catch {
            fputs("serve: failed to start: \(error)\n", stderr)
            exit(1)
        }
    }

    /// Programmatic entry point used by tests. Returns a `Server` handle that
    /// owns the bound socket + listener thread. Call `stop()` to release the
    /// port.
    public static func start(modelPath: String, host: String = "127.0.0.1",
                              port: UInt16 = 0, maxContext: Int? = nil) throws -> Server
    {
        return try Server.boot(modelPath: modelPath, host: host, port: port,
                                maxContextOverride: maxContext)
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt serve <model.tinygpt | hf-dir> [options]

        --port N              TCP port (default: 8080; 0 = pick any free port)
        --host HOST           bind address (default: 127.0.0.1)
        --max-context N       cap context length below the model's native limit
                              (useful when running lm-eval on long-prompt tasks
                               like MMLU-Pro where the harness sometimes overshoots)

        Endpoints:
          POST /v1/chat/completions   OpenAI ChatCompletion (messages: [...])
          POST /v1/completions        OpenAI Completion (prompt: "...")
          GET  /v1/models             list loaded model

        Designed for lm-evaluation-harness `local-chat-completions` adapter.
        Example: lm-eval --model local-chat-completions \\
                    --tasks hellaswag,arc_easy --model_args \\
                    "base_url=http://127.0.0.1:8080/v1/chat/completions,model=tinygpt"
        """)
        exit(2)
    }
}

// MARK: - Server

extension Serve {
    /// Handle to a running HTTP server. Reachable from tests + the CLI entry
    /// point. Owns the listener socket + the inference state.
    public final class Server: @unchecked Sendable {
        public let port: UInt16
        public let host: String
        public let config: ModelConfig
        public let maxContext: Int
        let model: AnyModel
        let tokenizer: TokenizerBox
        private let listenFd: Int32
        private let inferenceQueue: DispatchQueue
        private var running: Bool = true

        init(listenFd: Int32, host: String, port: UInt16,
             model: AnyModel, config: ModelConfig, tokenizer: TokenizerBox,
             maxContext: Int)
        {
            self.listenFd = listenFd
            self.host = host
            self.port = port
            self.model = model
            self.config = config
            self.tokenizer = tokenizer
            self.maxContext = maxContext
            self.inferenceQueue = DispatchQueue(label: "tinygpt.serve.inference")
        }

        static func boot(modelPath: String, host: String, port: UInt16,
                          maxContextOverride: Int?) throws -> Server
        {
            // Writes to a socket whose peer has hung up raise SIGPIPE,
            // which by default kills the process. SSE clients (curl
            // --max-time, browser fetch().cancel(), user closing a tab)
            // routinely close mid-stream, so we ignore SIGPIPE
            // process-wide and rely on write()'s EPIPE return + the
            // cancellation path in streamChat / streamCompletion.
            signal(SIGPIPE, SIG_IGN)

            // Load model + (optional) BPE tokenizer up front. Same logic as
            // Sample.swift so behaviour matches between `sample` and `serve`.
            let load = try ModelLoader.load(modelPath)
            let cfg = load.config
            let tok: TokenizerBox
            if let dir = load.hfTokenizerDir {
                do {
                    let hf = try HFTokenizer.loadBlocking(from: dir)
                    tok = .hf(hf)
                } catch {
                    fputs("warning: tokenizer load failed (\(error)); falling back to byte-level\n", stderr)
                    tok = .byteLevel
                }
            } else {
                tok = .byteLevel
            }

            let (fd, boundPort) = try Self.bindListener(host: host, port: port)
            let maxCtx = min(maxContextOverride ?? cfg.contextLength, cfg.contextLength)
            let server = Server(listenFd: fd, host: host, port: boundPort,
                                 model: load.model, config: cfg, tokenizer: tok,
                                 maxContext: maxCtx)
            server.startAcceptLoop()
            return server
        }

        /// Stops accepting new connections and closes the listening socket.
        /// In-flight requests on existing connections continue to completion.
        public func stop() {
            running = false
            Darwin.close(listenFd)
        }

        // MARK: BSD socket setup

        private static func bindListener(host: String, port: UInt16) throws -> (Int32, UInt16) {
            let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
            guard fd >= 0 else {
                throw NSError(domain: "tinygpt.serve", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "socket() failed: \(String(cString: strerror(errno)))"])
            }
            // SO_REUSEADDR — convenient when restarting the server quickly.
            var yes: Int32 = 1
            _ = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))

            var addr = sockaddr_in()
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = in_port_t(port).bigEndian
            // inet_pton: parse "127.0.0.1" into the in_addr struct.
            let inetResult = host.withCString { hostC -> Int32 in
                inet_pton(AF_INET, hostC, &addr.sin_addr)
            }
            guard inetResult == 1 else {
                Darwin.close(fd)
                throw NSError(domain: "tinygpt.serve", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "bad host: \(host)"])
            }
            let bindResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                    Darwin.bind(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard bindResult == 0 else {
                let err = String(cString: strerror(errno))
                Darwin.close(fd)
                throw NSError(domain: "tinygpt.serve", code: 3,
                              userInfo: [NSLocalizedDescriptionKey: "bind() failed: \(err)"])
            }
            guard listen(fd, 16) == 0 else {
                let err = String(cString: strerror(errno))
                Darwin.close(fd)
                throw NSError(domain: "tinygpt.serve", code: 4,
                              userInfo: [NSLocalizedDescriptionKey: "listen() failed: \(err)"])
            }
            // Resolve the actual bound port (port==0 → kernel-assigned).
            var bound = sockaddr_in()
            var bound_len = socklen_t(MemoryLayout<sockaddr_in>.size)
            let actualPort: UInt16
            let gotName = withUnsafeMutablePointer(to: &bound) { ptr -> Int32 in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                    getsockname(fd, sockPtr, &bound_len)
                }
            }
            if gotName == 0 {
                actualPort = UInt16(bigEndian: bound.sin_port)
            } else {
                actualPort = port
            }
            return (fd, actualPort)
        }

        // MARK: Accept loop

        private func startAcceptLoop() {
            Thread.detachNewThread { [weak self] in
                guard let self = self else { return }
                while self.running {
                    var clientAddr = sockaddr_in()
                    var clientLen = socklen_t(MemoryLayout<sockaddr_in>.size)
                    let clientFd = withUnsafeMutablePointer(to: &clientAddr) { ptr -> Int32 in
                        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                            accept(self.listenFd, sockPtr, &clientLen)
                        }
                    }
                    if clientFd < 0 {
                        if !self.running { return }
                        if errno == EINTR { continue }
                        // Listening socket closed — exit loop cleanly.
                        return
                    }
                    // One thread per connection. The inference is serialised
                    // on `inferenceQueue` regardless of how many connections
                    // there are.
                    Thread.detachNewThread { [weak self] in
                        guard let self = self else {
                            Darwin.close(clientFd); return
                        }
                        self.handleConnection(clientFd: clientFd)
                    }
                }
            }
        }

        // MARK: Per-connection handler

        private func handleConnection(clientFd: Int32) {
            defer { Darwin.close(clientFd) }
            guard let request = HTTPRequest.read(from: clientFd) else {
                respond(clientFd: clientFd, status: 400, body: "bad request")
                return
            }

            // Health check / model listing.
            if request.method == "GET" && request.path == "/v1/models" {
                let payload: [String: Any] = [
                    "object": "list",
                    "data": [
                        ["id": "tinygpt", "object": "model", "owned_by": "tinygpt"]
                    ]
                ]
                respondJSON(clientFd: clientFd, status: 200, payload: payload)
                return
            }

            if request.method == "POST" && request.path == "/v1/chat/completions" {
                handleChatCompletions(clientFd: clientFd, body: request.body)
                return
            }
            if request.method == "POST" && request.path == "/v1/completions" {
                handleCompletions(clientFd: clientFd, body: request.body)
                return
            }
            respond(clientFd: clientFd, status: 404, body: "not found: \(request.method) \(request.path)")
        }

        // MARK: /v1/chat/completions

        private func handleChatCompletions(clientFd: Int32, body: Data) {
            do {
                guard let json = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                    respond(clientFd: clientFd, status: 400, body: "json must be object"); return
                }
                let messages = (json["messages"] as? [[String: Any]]) ?? []
                let prompt = renderChatMessages(messages)
                let maxTokens = (json["max_tokens"] as? Int) ?? 128
                let temperature = (json["temperature"] as? Double).map { Float($0) } ?? 0.0
                let stopParam = readStopParam(json["stop"])
                let stream = (json["stream"] as? Bool) ?? false

                if stream {
                    streamChat(clientFd: clientFd, prompt: prompt,
                                maxTokens: maxTokens, temperature: temperature,
                                stop: stopParam)
                    return
                }

                let (text, promptTokens, completionTokens) = try inferenceQueue.sync {
                    try self.generate(prompt: prompt, maxTokens: maxTokens,
                                       temperature: temperature, stop: stopParam)
                }

                let payload: [String: Any] = [
                    "id": "chatcmpl-\(UUID().uuidString)",
                    "object": "chat.completion",
                    "created": Int(Date().timeIntervalSince1970),
                    "model": "tinygpt",
                    "choices": [[
                        "index": 0,
                        "message": ["role": "assistant", "content": text],
                        "finish_reason": "stop"
                    ]],
                    "usage": [
                        "prompt_tokens": promptTokens,
                        "completion_tokens": completionTokens,
                        "total_tokens": promptTokens + completionTokens
                    ]
                ]
                respondJSON(clientFd: clientFd, status: 200, payload: payload)
            } catch {
                respond(clientFd: clientFd, status: 500, body: "error: \(error)")
            }
        }

        // MARK: /v1/completions

        private func handleCompletions(clientFd: Int32, body: Data) {
            do {
                guard let json = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                    respond(clientFd: clientFd, status: 400, body: "json must be object"); return
                }
                let prompt = (json["prompt"] as? String) ?? ""
                let maxTokens = (json["max_tokens"] as? Int) ?? 128
                let temperature = (json["temperature"] as? Double).map { Float($0) } ?? 0.0
                let stopParam = readStopParam(json["stop"])
                let stream = (json["stream"] as? Bool) ?? false

                if stream {
                    streamCompletion(clientFd: clientFd, prompt: prompt,
                                      maxTokens: maxTokens, temperature: temperature,
                                      stop: stopParam)
                    return
                }

                let (text, promptTokens, completionTokens) = try inferenceQueue.sync {
                    try self.generate(prompt: prompt, maxTokens: maxTokens,
                                       temperature: temperature, stop: stopParam)
                }

                let payload: [String: Any] = [
                    "id": "cmpl-\(UUID().uuidString)",
                    "object": "text_completion",
                    "created": Int(Date().timeIntervalSince1970),
                    "model": "tinygpt",
                    "choices": [[
                        "index": 0,
                        "text": text,
                        "finish_reason": "stop"
                    ]],
                    "usage": [
                        "prompt_tokens": promptTokens,
                        "completion_tokens": completionTokens,
                        "total_tokens": promptTokens + completionTokens
                    ]
                ]
                respondJSON(clientFd: clientFd, status: 200, payload: payload)
            } catch {
                respond(clientFd: clientFd, status: 500, body: "error: \(error)")
            }
        }

        // MARK: SSE streaming

        /// Streaming variant of /v1/chat/completions.
        ///
        /// Wire format (OpenAI-compatible):
        ///   data: {"id":"chatcmpl-...","object":"chat.completion.chunk",
        ///           "created":TS,"model":"tinygpt",
        ///           "choices":[{"index":0,"delta":{"role":"assistant"}}]}
        ///
        ///   data: {...,"choices":[{"index":0,"delta":{"content":"hello"}}]}
        ///   data: {...,"choices":[{"index":0,"delta":{"content":" world"}}]}
        ///   ...
        ///   data: {...,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
        ///   data: [DONE]
        ///
        /// One generation token may not produce visible output (partial
        /// BPE byte) — we only emit a delta when the running decoded
        /// suffix grows. This matches OpenAI's behavior on multi-byte
        /// tokens (Chinese, emoji, etc.) and is what SSE clients expect.
        private func streamChat(clientFd: Int32, prompt: String,
                                  maxTokens: Int, temperature: Float,
                                  stop: [String]) {
            let id = "chatcmpl-\(UUID().uuidString)"
            writeSSEHead(clientFd: clientFd)
            // Opening delta — sets role on the assistant message.
            writeSSEEvent(clientFd: clientFd, payload: chunkPayload(
                id: id, object: "chat.completion.chunk",
                delta: ["role": "assistant"], finishReason: nil))

            var finishReason = "stop"
            var clientGone = false
            do {
                try inferenceQueue.sync {
                    try self.generateStreaming(prompt: prompt, maxTokens: maxTokens,
                                                temperature: temperature, stop: stop)
                    { newText in
                        let ok = self.writeSSEEvent(clientFd: clientFd, payload: self.chunkPayload(
                            id: id, object: "chat.completion.chunk",
                            delta: ["content": newText], finishReason: nil))
                        if !ok { clientGone = true }
                        return ok
                    }
                }
            } catch {
                finishReason = "error"
            }
            // Client disconnected — don't bother sending final chunk + DONE,
            // the socket is dead. Just return so the connection closes.
            if clientGone { return }
            // Final delta with finish_reason — empty delta per OpenAI spec.
            writeSSEEvent(clientFd: clientFd, payload: chunkPayload(
                id: id, object: "chat.completion.chunk",
                delta: [:], finishReason: finishReason))
            writeSSETerminator(clientFd: clientFd)
        }

        /// Streaming variant of /v1/completions. Same wire as streamChat
        /// but uses "text_completion" object type and `text` field in the
        /// choice instead of `delta`.
        private func streamCompletion(clientFd: Int32, prompt: String,
                                        maxTokens: Int, temperature: Float,
                                        stop: [String]) {
            let id = "cmpl-\(UUID().uuidString)"
            writeSSEHead(clientFd: clientFd)

            var finishReason = "stop"
            var clientGone = false
            do {
                try inferenceQueue.sync {
                    try self.generateStreaming(prompt: prompt, maxTokens: maxTokens,
                                                temperature: temperature, stop: stop)
                    { newText in
                        let payload: [String: Any] = [
                            "id": id,
                            "object": "text_completion",
                            "created": Int(Date().timeIntervalSince1970),
                            "model": "tinygpt",
                            "choices": [[
                                "index": 0,
                                "text": newText,
                                "finish_reason": NSNull()
                            ]]
                        ]
                        let ok = self.writeSSEEvent(clientFd: clientFd, payload: payload)
                        if !ok { clientGone = true }
                        return ok
                    }
                }
            } catch {
                finishReason = "error"
            }
            if clientGone { return }
            let final: [String: Any] = [
                "id": id,
                "object": "text_completion",
                "created": Int(Date().timeIntervalSince1970),
                "model": "tinygpt",
                "choices": [[
                    "index": 0,
                    "text": "",
                    "finish_reason": finishReason
                ]]
            ]
            writeSSEEvent(clientFd: clientFd, payload: final)
            writeSSETerminator(clientFd: clientFd)
        }

        private func chunkPayload(id: String, object: String,
                                   delta: [String: Any],
                                   finishReason: String?) -> [String: Any] {
            var choice: [String: Any] = [
                "index": 0,
                "delta": delta
            ]
            choice["finish_reason"] = finishReason ?? NSNull()
            return [
                "id": id,
                "object": object,
                "created": Int(Date().timeIntervalSince1970),
                "model": "tinygpt",
                "choices": [choice]
            ]
        }

        /// Streaming generation. Calls `onText` with the newly-decoded
        /// suffix each time a step extends the visible string. The
        /// callback returns `true` to continue, `false` to abort (used
        /// to propagate client-disconnect through to early exit). The
        /// token loop is the same as `generate(...)` — extracted into a
        /// callback-driven variant rather than duplicated.
        func generateStreaming(prompt: String, maxTokens: Int,
                                temperature: Float, stop: [String],
                                onText: (String) -> Bool) throws
        {
            let promptIds = tokenizer.encode(prompt)
            if promptIds.isEmpty { return }
            let promptArr = MLXArray(promptIds.map { Int32($0) }, [1, promptIds.count])
            let ctxCap = maxContext
            let bounded: MLXArray
            if promptArr.shape[1] > ctxCap {
                bounded = promptArr[0..., (promptArr.shape[1] - ctxCap) ..< promptArr.shape[1]]
            } else {
                bounded = promptArr
            }
            var idx = bounded
            var generated: [Int] = []
            generated.reserveCapacity(maxTokens)
            var lastDecoded: String = ""
            for _ in 0..<maxTokens {
                let T = idx.shape.last!
                let lo = max(0, T - ctxCap)
                let cond = idx[0..., lo..<T]
                let logits = model(cond)
                let last = logits[0..., logits.shape[1] - 1, 0...]
                let nextId: MLXArray
                if temperature <= 0 {
                    nextId = argMax(last, axis: -1).reshaped([1, 1])
                } else {
                    let scaled = last / MLXArray(temperature)
                    nextId = MLXRandom.categorical(scaled).reshaped([1, 1])
                }
                eval(nextId)
                generated.append(Int(nextId.item(Int32.self)))

                let nowDecoded = tokenizer.decode(generated)
                // Only emit a delta when the visible string grew. BPE /
                // multi-byte UTF-8 can leave us with intermediate bytes
                // that don't yet form a complete character.
                if nowDecoded.count > lastDecoded.count
                    && nowDecoded.hasPrefix(lastDecoded) {
                    let suffix = String(nowDecoded.dropFirst(lastDecoded.count))
                    let keepGoing = onText(suffix)
                    lastDecoded = nowDecoded
                    if !keepGoing { return }  // client disconnected — abort early
                }

                if !stop.isEmpty {
                    if stop.contains(where: { !$0.isEmpty && nowDecoded.contains($0) }) {
                        return
                    }
                }
                idx = concatenated([idx, nextId.asType(idx.dtype)], axis: 1)
            }
        }

        @discardableResult
        private func writeSSEHead(clientFd: Int32) -> Bool {
            var head = "HTTP/1.1 200 OK\r\n"
            head += "Content-Type: text/event-stream; charset=utf-8\r\n"
            head += "Cache-Control: no-cache\r\n"
            head += "Connection: close\r\n"
            head += "X-Accel-Buffering: no\r\n"  // tell reverse-proxies not to buffer
            head += "\r\n"
            return writeAll(clientFd: clientFd, data: Data(head.utf8))
        }

        /// Returns true if the event was delivered, false if the client
        /// has disconnected. Streaming endpoints use the return value to
        /// short-circuit the generation loop when the user aborts.
        @discardableResult
        private func writeSSEEvent(clientFd: Int32, payload: [String: Any]) -> Bool {
            guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return false }
            var frame = "data: ".data(using: .utf8)!
            frame.append(data)
            frame.append(Data("\n\n".utf8))
            return writeAll(clientFd: clientFd, data: frame)
        }

        @discardableResult
        private func writeSSETerminator(clientFd: Int32) -> Bool {
            return writeAll(clientFd: clientFd, data: Data("data: [DONE]\n\n".utf8))
        }

        private func readStopParam(_ raw: Any?) -> [String] {
            if let s = raw as? String { return [s] }
            if let arr = raw as? [String] { return arr }
            return []
        }

        // MARK: Generate

        /// Encode the prompt, run an uncached generation loop (we don't keep
        /// state across HTTP calls), decode the completion, return text +
        /// token counts. Single-call generation — KV cache is built fresh
        /// each call because the harness sends independent prompts.
        func generate(prompt: String, maxTokens: Int, temperature: Float,
                       stop: [String]) throws -> (String, Int, Int)
        {
            let promptIds = tokenizer.encode(prompt)
            if promptIds.isEmpty {
                return ("", 0, 0)
            }
            let promptArr = MLXArray(promptIds.map { Int32($0) }, [1, promptIds.count])

            // Bound context: drop the head of the prompt if it overflows.
            // lm-eval's MMLU-Pro prompts can hit ~3K tokens — beyond our 256
            // default contextLength. For 0-shot tasks we cope by truncating
            // from the left so the most relevant tail survives.
            let ctxCap = maxContext
            let bounded: MLXArray
            if promptArr.shape[1] > ctxCap {
                bounded = promptArr[0..., (promptArr.shape[1] - ctxCap) ..< promptArr.shape[1]]
            } else {
                bounded = promptArr
            }

            var idx = bounded
            var generated: [Int] = []
            generated.reserveCapacity(maxTokens)
            for _ in 0..<maxTokens {
                let T = idx.shape.last!
                let lo = max(0, T - ctxCap)
                let cond = idx[0..., lo..<T]
                let logits = model(cond)
                let last = logits[0..., logits.shape[1] - 1, 0...]
                let nextId: MLXArray
                if temperature <= 0 {
                    nextId = argMax(last, axis: -1).reshaped([1, 1])
                } else {
                    let scaled = last / MLXArray(temperature)
                    nextId = MLXRandom.categorical(scaled).reshaped([1, 1])
                }
                eval(nextId)
                let tokenInt = Int(nextId.item(Int32.self))
                generated.append(tokenInt)

                // Stop-string detection: re-decode the running tail and check
                // whether any user-supplied stop string appears. We slice the
                // *decoded* output rather than ids because BPE tokens don't
                // align with characters.
                if !stop.isEmpty {
                    let renderedSoFar = tokenizer.decode(generated)
                    if stop.contains(where: { !$0.isEmpty && renderedSoFar.contains($0) }) {
                        // Trim everything from (and including) the stop string.
                        if let trimmed = trimAtStop(renderedSoFar, stops: stop) {
                            return (trimmed, promptIds.count, generated.count)
                        }
                    }
                }
                idx = concatenated([idx, nextId.asType(idx.dtype)], axis: 1)
            }
            let text = tokenizer.decode(generated)
            return (text, promptIds.count, generated.count)
        }

        private func trimAtStop(_ text: String, stops: [String]) -> String? {
            // Find the earliest matching stop in the decoded text and trim there.
            var earliest: String.Index? = nil
            for s in stops where !s.isEmpty {
                if let r = text.range(of: s) {
                    if earliest == nil || r.lowerBound < earliest! {
                        earliest = r.lowerBound
                    }
                }
            }
            guard let cut = earliest else { return nil }
            return String(text[..<cut])
        }

        /// OpenAI ChatCompletion `messages` → flat prompt. We use ChatML-ish
        /// formatting because that's what tinygpt's SFT templates can match;
        /// if your model was trained on a different template, prefer the
        /// `/v1/completions` endpoint and pass an already-formatted prompt.
        private func renderChatMessages(_ messages: [[String: Any]]) -> String {
            var out = ""
            for m in messages {
                let role = (m["role"] as? String) ?? "user"
                let content = (m["content"] as? String) ?? ""
                out += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
            }
            out += "<|im_start|>assistant\n"
            return out
        }

        // MARK: HTTP response helpers

        private func respond(clientFd: Int32, status: Int, body: String) {
            let statusText = httpStatusText(status)
            let bytes = [UInt8](body.utf8)
            var head = "HTTP/1.1 \(status) \(statusText)\r\n"
            head += "Content-Type: text/plain; charset=utf-8\r\n"
            head += "Content-Length: \(bytes.count)\r\n"
            head += "Connection: close\r\n"
            head += "\r\n"
            writeAll(clientFd: clientFd, data: Data(head.utf8))
            writeAll(clientFd: clientFd, data: Data(bytes))
        }

        private func respondJSON(clientFd: Int32, status: Int, payload: [String: Any]) {
            let data: Data
            do {
                data = try JSONSerialization.data(withJSONObject: payload)
            } catch {
                respond(clientFd: clientFd, status: 500, body: "json encode failed: \(error)")
                return
            }
            var head = "HTTP/1.1 \(status) \(httpStatusText(status))\r\n"
            head += "Content-Type: application/json; charset=utf-8\r\n"
            head += "Content-Length: \(data.count)\r\n"
            head += "Connection: close\r\n"
            head += "\r\n"
            writeAll(clientFd: clientFd, data: Data(head.utf8))
            writeAll(clientFd: clientFd, data: data)
        }

        /// Write all bytes to the client socket. Returns true on success,
        /// false if the peer disconnected mid-write (write returned 0 or
        /// -1 with errno=EPIPE/ECONNRESET). Callers that care about
        /// peer-gone (the streaming endpoints) check this to abort the
        /// generation loop early.
        @discardableResult
        private func writeAll(clientFd: Int32, data: Data) -> Bool {
            var ok = true
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                guard let base = raw.baseAddress else { ok = false; return }
                var sent = 0
                while sent < data.count {
                    let n = Darwin.write(clientFd, base.advanced(by: sent), data.count - sent)
                    if n <= 0 { ok = false; return }
                    sent += n
                }
            }
            return ok
        }

        private func httpStatusText(_ code: Int) -> String {
            switch code {
            case 200: return "OK"
            case 400: return "Bad Request"
            case 404: return "Not Found"
            case 500: return "Internal Server Error"
            default:  return "Status"
            }
        }
    }

    // MARK: Tokenizer wrapper

    /// `TokenizerBox` lets `Server` work with either ByteTokenizer or
    /// HFTokenizer without the rest of the file caring. Same shape as
    /// Sample.swift's logic, factored for reuse.
    public enum TokenizerBox: @unchecked Sendable {
        case byteLevel
        case hf(HFTokenizer)

        func encode(_ text: String) -> [Int] {
            switch self {
            case .byteLevel: return [UInt8](text.utf8).map { Int($0) }
            case .hf(let t):
                do { return try t.encode(text) }
                catch { return [] }
            }
        }
        func decode(_ ids: [Int]) -> String {
            switch self {
            case .byteLevel:
                let bytes = ids.compactMap { (id: Int) -> UInt8? in
                    guard id >= 0 && id < 256 else { return nil }
                    return UInt8(id)
                }
                return String(decoding: bytes, as: UTF8.self)
            case .hf(let t):
                return t.decode(ids)
            }
        }
    }
}

// MARK: - HTTP request parser
//
// Hand-rolled because we want zero deps. Reads the request line, header
// block (until "\r\n\r\n"), then drains Content-Length bytes of body.
// No chunked encoding (lm-eval-harness doesn't use it).

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    static func read(from fd: Int32) -> HTTPRequest? {
        var buf = Data()
        var chunk = [UInt8](repeating: 0, count: 4096)
        // Read until we see the end of the header block.
        while true {
            let n = chunk.withUnsafeMutableBufferPointer { ptr -> Int in
                Darwin.read(fd, ptr.baseAddress, ptr.count)
            }
            if n <= 0 { return nil }
            buf.append(chunk, count: n)
            if let _ = buf.range(of: Data("\r\n\r\n".utf8)) {
                break
            }
            if buf.count > 16 * 1024 * 1024 { return nil }
        }
        guard let headerEnd = buf.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let headerData = buf.prefix(upTo: headerEnd.lowerBound)
        let headerText = String(decoding: headerData, as: UTF8.self)
        let lines = headerText.split(separator: "\r\n", omittingEmptySubsequences: false)
        guard let firstLine = lines.first else { return nil }
        let parts = firstLine.split(separator: " ", maxSplits: 2)
        guard parts.count >= 2 else { return nil }
        let method = String(parts[0])
        let path = String(parts[1])
        var headers: [String: String] = [:]
        for line in lines.dropFirst() where !line.isEmpty {
            if let colon = line.firstIndex(of: ":") {
                let k = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
                let v = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
                headers[k] = v
            }
        }

        let contentLength = Int(headers["content-length"] ?? "0") ?? 0
        var body = buf.suffix(from: headerEnd.upperBound)
        while body.count < contentLength {
            let n = chunk.withUnsafeMutableBufferPointer { ptr -> Int in
                Darwin.read(fd, ptr.baseAddress, ptr.count)
            }
            if n <= 0 { break }
            body.append(chunk, count: n)
        }
        // Trim if we read past Content-Length (extra pipelined bytes).
        if body.count > contentLength {
            body = body.prefix(contentLength)
        }
        return HTTPRequest(method: method, path: path, headers: headers, body: Data(body))
    }
}
