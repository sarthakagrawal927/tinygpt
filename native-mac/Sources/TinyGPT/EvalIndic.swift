import Foundation
import MLX
import TinyGPTData
import TinyGPTIO
import TinyGPTModel

/// `tinygpt eval-indic` — Indic-language eval harness for MILU and
/// IndicGenBench. This is the Wave 4 doorway for claiming Hindi /
/// multilingual support: before any specialist training, run base
/// models through these two evals to get a real baseline.
///
/// The two evals:
///   - **MILU** (AI4Bharat, NAACL 2025) — MMLU-style multiple choice
///     across 11 Indic languages × 41 subjects. We score the standard
///     way: for each question, pick the option with the highest
///     log-likelihood under the model. Metric: accuracy.
///     Paper: arXiv 2411.02538. Repo: github.com/AI4Bharat/MILU.
///
///   - **IndicGenBench** (Google Research, 2024) — generative tasks
///     across 29 Indic languages. We wire one subtask here, IndicXQuAD
///     (extractive QA), as the smoke test: greedy-generate the answer
///     span, score by exact-match against the gold span.
///     Paper: arXiv 2404.16816. Repo:
///     github.com/google-research-datasets/indic-gen-bench.
///
/// USAGE
///
///   # MILU (Hindi split, 100 samples)
///   tinygpt eval-indic --task milu --model /tmp/flagship-huge.tinygpt \
///       --milu-data ~/.cache/tinygpt/datasets/ai4bharat/MILU/hi.jsonl \
///       --limit 100
///
///   # IndicGenBench XQuAD (Hindi split, 50 samples)
///   tinygpt eval-indic --task indicgenbench --subtask xquad \
///       --model /tmp/flagship-huge.tinygpt \
///       --indicgen-data ~/.cache/tinygpt/datasets/google/IndicGenBench_xquad_in/hi.jsonl \
///       --limit 50
///
///   # Both, aggregate report
///   tinygpt eval-indic --task all --model /tmp/flagship-huge.tinygpt \
///       --milu-data … --indicgen-data … --limit 100
///
/// DATA PREP
///
/// Datasets are not bundled. Pre-fetch with:
///
///   tinygpt download-dataset ai4bharat/MILU --format plain
///   # ↑ produces ~/.cache/tinygpt/datasets/ai4bharat/MILU/*.jsonl
///   # For IndicGenBench:
///   tinygpt download-dataset google/IndicGenBench_xquad_in --format plain
///
/// Or pass any local JSONL with the expected schemas (see SCHEMAS below).
///
/// SCHEMAS
///
/// MILU JSONL row:
///   {
///     "question": "<hindi text>",
///     "option1": "<text>", "option2": "<text>",
///     "option3": "<text>", "option4": "<text>",
///     "answer": "option2",        // or "B" / 2 / "option2 text" — we try each
///     "language": "Hindi",        // optional, for filtering
///     "subject": "History"        // optional
///   }
///
/// IndicGenBench XQuAD row:
///   {
///     "question": "<question text>",
///     "context": "<paragraph>",
///     "answers": { "text": ["gold answer"], "answer_start": [int] },
///     "language": "hi"            // optional
///   }
///
/// SCORING
///
/// - MILU: argmax over option log-likelihoods under the model.
///   Equivalent to lm-eval-harness's `multiple_choice` task type.
/// - IndicXQuAD: greedy generation with a max-tokens cap of 32. Exact
///   match against any gold answer in `answers.text[]`, normalized
///   (lowercased, stripped punctuation).
///
/// LIMITATIONS (current shipping state — see docs/research/indic_evals.md)
///
/// 1. **Byte-level models score near 0**: a Latin-only byte tokenizer
///    fragments Devanagari into 3-byte UTF-8 sequences. The model has
///    never seen these byte n-grams during training (Shakespeare /
///    English corpora). Expect near-random or zero on MILU/IndicXQuAD.
///    This IS the baseline the doc reports.
/// 2. **No few-shot prompting yet**: the standard MILU protocol uses
///    5-shot. We do zero-shot. Score gap with paper numbers ~ 5–8pts.
/// 3. **Single-language at a time**: aggregate across 11 langs requires
///    one invocation per language file. Wrap with shell loop.
/// 4. **No log-likelihood batching**: each option is scored serially.
///    Acceptable for --limit 100; for full MILU (~85k questions × 4
///    options × 11 languages = 3.7M forwards) you want batched scoring.
enum EvalIndic {

    // MARK: - CLI entry

    static func run(args: [String]) {
        var task: String? = nil
        var subtask = "xquad"
        var modelPath: String? = nil
        var miluData: String? = nil
        var indicgenData: String? = nil
        var limit: Int? = nil
        var maxNewTokens = 32
        var verbose = false

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--task":           task = args[i+1]; i += 2
            case "--subtask":        subtask = args[i+1]; i += 2
            case "--model":          modelPath = args[i+1]; i += 2
            case "--milu-data":      miluData = args[i+1]; i += 2
            case "--indicgen-data":  indicgenData = args[i+1]; i += 2
            case "--limit":          limit = Int(args[i+1]); i += 2
            case "--max-new-tokens": maxNewTokens = Int(args[i+1]) ?? maxNewTokens; i += 2
            case "--verbose":        verbose = true; i += 1
            case "-h", "--help":     exitUsage()
            default:
                if args[i].hasPrefix("-") {
                    fputs("eval-indic: unknown flag: \(args[i])\n", stderr); exitUsage()
                }
                // Allow positional model path as a convenience.
                if modelPath == nil { modelPath = args[i] }
                i += 1
            }
        }

        guard let task = task else {
            fputs("eval-indic: --task is required (milu | indicgenbench | all)\n", stderr)
            exitUsage()
        }
        guard let modelPath = modelPath else {
            fputs("eval-indic: --model is required\n", stderr); exitUsage()
        }

        // Header.
        print("""

        TinyGPT — eval-indic
        --------------------
        task:    \(task)\(task == "indicgenbench" ? " (\(subtask))" : "")
        model:   \(modelPath)
        limit:   \(limit.map(String.init) ?? "all")
        """)

        // Load the model once and reuse across tasks.
        let load: ModelLoader.LoadResult
        do { load = try ModelLoader.load(modelPath) }
        catch {
            fputs("eval-indic: model load failed: \(error)\n", stderr); exit(1)
        }
        let model = load.model
        let cfg = load.config
        let isBpe = (load.hfTokenizerDir != nil)
        print("model:   \(cfg.nLayers)L · d=\(cfg.dModel) · ctx=\(cfg.contextLength) · vocab=\(cfg.vocabSize) · \(isBpe ? "BPE" : "byte-level")")

        if !isBpe {
            print("""

            ⚠  byte-level model detected. Indic-language eval against a byte-
               level tokenizer fragments Devanagari into UTF-8 byte triples
               the model has never seen. Expect ~0 accuracy on MILU and ~0
               exact-match on IndicXQuAD. This is the documented baseline
               — see docs/research/indic_evals.md for context.
            """)
        }

        // Build a tokenizer once if we'll need it (BPE path).
        var tokenizer: HFTokenizer? = nil
        if let tokDir = load.hfTokenizerDir {
            do { tokenizer = try HFTokenizer.loadBlocking(from: tokDir) }
            catch {
                fputs("eval-indic: tokenizer load failed: \(error)\n", stderr); exit(1)
            }
        }

        var milu: TaskResult? = nil
        var indic: TaskResult? = nil

        if task == "milu" || task == "all" {
            guard let path = miluData else {
                fputs("""
                eval-indic: --milu-data <path.jsonl> is required for task=milu
                  Try:
                    tinygpt download-dataset ai4bharat/MILU
                  then point --milu-data at the resulting JSONL.
                  See docs/research/indic_evals.md for the schema.
                """, stderr)
                exit(2)
            }
            milu = runMILU(path: path, model: model, cfg: cfg,
                           tokenizer: tokenizer, limit: limit, verbose: verbose)
        }

        if task == "indicgenbench" || task == "all" {
            guard let path = indicgenData else {
                fputs("""
                eval-indic: --indicgen-data <path.jsonl> is required for task=indicgenbench
                  Try:
                    tinygpt download-dataset google/IndicGenBench_xquad_in
                  then point --indicgen-data at the resulting JSONL.
                  See docs/research/indic_evals.md for the schema.
                """, stderr)
                exit(2)
            }
            indic = runIndicGenBench(path: path, subtask: subtask,
                                     model: model, cfg: cfg,
                                     tokenizer: tokenizer,
                                     limit: limit,
                                     maxNewTokens: maxNewTokens,
                                     verbose: verbose)
        }

        if task != "milu" && task != "indicgenbench" && task != "all" {
            fputs("eval-indic: unknown --task '\(task)' (expected milu | indicgenbench | all)\n", stderr)
            exit(2)
        }

        // Aggregate summary.
        print("""

        AGGREGATE
        ---------
        """)
        if let r = milu {
            print(String(format: "  MILU            accuracy = %5.2f%%  (%d / %d)",
                         r.scorePct, r.correct, r.total))
        }
        if let r = indic {
            print(String(format: "  IndicGenBench   \(subtask) EM = %5.2f%%  (%d / %d)",
                         r.scorePct, r.correct, r.total))
        }
        if milu == nil && indic == nil {
            print("  (nothing scored)")
        }
        print("")
    }

    // MARK: - Result shape

    private struct TaskResult {
        let total: Int
        let correct: Int
        var scorePct: Double {
            total == 0 ? 0 : 100.0 * Double(correct) / Double(total)
        }
    }

    // MARK: - MILU

    /// Score MILU rows. For each row, encode (prompt + each option) and
    /// pick the option with the lowest cross-entropy (= highest log-
    /// likelihood). This is the standard `multiple_choice` adapter that
    /// lm-eval-harness uses; we do it inline because the harness adapter
    /// can't easily ingest Indic-script MILU shards (no canonical lm-eval
    /// task definition yet).
    private static func runMILU(path: String, model: AnyModel,
                                cfg: ModelConfig, tokenizer: HFTokenizer?,
                                limit: Int?, verbose: Bool) -> TaskResult {
        print("\n→ MILU")
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("  ! data file not found: \(path)")
            print("    Pre-fetch with: tinygpt download-dataset ai4bharat/MILU")
            return TaskResult(total: 0, correct: 0)
        }

        // Iterate JSONL; accept either MILU's canonical schema or the
        // generic "question/optionN/answer" shape.
        var seen = 0
        var correct = 0
        var skipped = 0
        do {
            _ = try RowReader.readRows(url: url, format: .jsonl) { row in
                if let cap = limit, seen >= cap { return false }
                guard let q = (row["question"] as? String) else {
                    skipped += 1; return true
                }
                let opts = extractMILUOptions(row: row)
                guard opts.count >= 2 else { skipped += 1; return true }
                guard let gold = extractMILUGoldIndex(row: row, options: opts) else {
                    skipped += 1; return true
                }
                let pickedIdx = scoreMCQOptions(model: model, cfg: cfg,
                                                tokenizer: tokenizer,
                                                question: q, options: opts)
                seen += 1
                if pickedIdx == gold { correct += 1 }
                if verbose || seen % 25 == 0 {
                    let acc = 100.0 * Double(correct) / Double(seen)
                    fputs(String(format: "  [%4d]  picked=%d  gold=%d  acc=%.2f%%\n",
                                  seen, pickedIdx, gold, acc), stderr)
                }
                return true
            }
        } catch {
            print("  ! read error: \(error)")
            return TaskResult(total: seen, correct: correct)
        }
        let acc = seen == 0 ? 0 : 100.0 * Double(correct) / Double(seen)
        print(String(format: "  result: %.2f%% (%d / %d)\(skipped > 0 ? "  (\(skipped) malformed rows skipped)" : "")",
                     acc, correct, seen))
        return TaskResult(total: seen, correct: correct)
    }

    /// Collect the option strings from a MILU row. MILU's HF shards use
    /// keys like option1..option4; some splits use options as an array.
    private static func extractMILUOptions(row: [String: Any]) -> [String] {
        // Option-array form: { "options": ["a","b","c","d"] }
        if let arr = row["options"] as? [Any] {
            return arr.compactMap { $0 as? String }
        }
        // option1..option5 form.
        var out: [String] = []
        for k in 1...10 {
            let key = "option\(k)"
            if let s = row[key] as? String, !s.isEmpty { out.append(s) }
        }
        if !out.isEmpty { return out }
        // a/b/c/d form (uppercase or lowercase).
        let letters = ["A", "B", "C", "D", "E"]
        for L in letters {
            if let s = (row[L] as? String) ?? (row[L.lowercased()] as? String) {
                out.append(s)
            }
        }
        return out
    }

    /// Map MILU's "answer" field to a 0-indexed option index. Robust to
    /// the four common encodings: index ("1"), letter ("B"), key
    /// ("option2"), or the literal option text.
    private static func extractMILUGoldIndex(row: [String: Any],
                                              options: [String]) -> Int? {
        // Numeric answer (could be 0-indexed or 1-indexed in the wild;
        // MILU paper convention is 1-indexed but we try both).
        if let n = row["answer"] as? Int {
            if n >= 1 && n <= options.count { return n - 1 }
            if n >= 0 && n < options.count { return n }
        }
        if let n = row["answer_index"] as? Int, n >= 0, n < options.count {
            return n
        }
        if let s = (row["answer"] as? String) {
            let stripped = s.trimmingCharacters(in: .whitespacesAndNewlines)
            // "B" / "b" — letter form.
            if stripped.count == 1, let scalar = stripped.uppercased().unicodeScalars.first {
                let code = Int(scalar.value)
                let aCode = Int(("A" as UnicodeScalar).value)
                let idx = code - aCode
                if idx >= 0 && idx < options.count { return idx }
            }
            // "option2" / "Option 3"
            let lower = stripped.lowercased().replacingOccurrences(of: " ", with: "")
            if lower.hasPrefix("option"), let n = Int(lower.dropFirst("option".count)) {
                if n >= 1 && n <= options.count { return n - 1 }
            }
            // Numeric string.
            if let n = Int(stripped) {
                if n >= 1 && n <= options.count { return n - 1 }
                if n >= 0 && n < options.count { return n }
            }
            // Literal answer text — match against options.
            for (i, opt) in options.enumerated() where opt == stripped {
                return i
            }
        }
        return nil
    }

    // MARK: - MCQ log-likelihood scoring

    /// Score multiple-choice by per-option log-likelihood.
    ///
    /// For each option we form the continuation `<question>\nAnswer: <option>`,
    /// score the *option* portion's tokens under the model (CE averaged
    /// over option tokens only), and return the argmin (lowest loss =
    /// highest log-likelihood). Length-normalised CE is the right
    /// scoring rule when options differ in length — without it the
    /// shortest option wins on prior.
    private static func scoreMCQOptions(model: AnyModel, cfg: ModelConfig,
                                         tokenizer: HFTokenizer?,
                                         question: String,
                                         options: [String]) -> Int {
        // Match the MMLU 0-shot template the lm-eval harness uses.
        let preamble = "Question: \(question)\nAnswer: "
        var bestIdx = 0
        var bestLoss = Float.greatestFiniteMagnitude
        for (i, opt) in options.enumerated() {
            let prefixIds = encode(text: preamble, tokenizer: tokenizer)
            let optIds    = encode(text: opt,      tokenizer: tokenizer)
            if optIds.isEmpty { continue }
            // Concatenate; we'll score only the option-positions' CE.
            let fullIds = prefixIds + optIds
            // Truncate from the LEFT so the option stays in-context. The
            // model's contextLength bounds the joint sequence.
            let ctx = cfg.contextLength
            let trimmed: [Int32]
            if fullIds.count > ctx {
                let lo = fullIds.count - ctx
                trimmed = Array(fullIds[lo..<fullIds.count]).map { Int32($0) }
            } else {
                trimmed = fullIds.map { Int32($0) }
            }
            // Inputs are tokens 0..<N-1; targets are tokens 1..<N.
            if trimmed.count < 2 { continue }
            let xIds = Array(trimmed.dropLast())
            let yIds = Array(trimmed.dropFirst())
            // Build a mask: only score positions whose target falls inside
            // the option suffix. The option suffix starts at index
            // (prefixIds.count - drop) in fullIds where drop = leftTrim.
            let leftTrim = max(0, fullIds.count - ctx)
            let optStartInTrimmed = max(0, prefixIds.count - leftTrim)
            var mask = [Float](repeating: 0, count: yIds.count)
            for j in 0..<yIds.count {
                // yIds[j] corresponds to predicting trimmed[j+1].
                if (j + 1) >= optStartInTrimmed { mask[j] = 1.0 }
            }
            if mask.reduce(0, +) < 1.0 { continue }

            let x = MLXArray(xIds, [1, xIds.count])
            let y = MLXArray(yIds, [1, yIds.count])
            let m = MLXArray(mask, [1, mask.count])
            let loss = model.maskedLoss(x, y, m)
            eval(loss)
            let lv = loss.item(Float.self)
            if lv < bestLoss {
                bestLoss = lv
                bestIdx = i
            }
        }
        return bestIdx
    }

    // MARK: - IndicGenBench / XQuAD

    /// Score IndicXQuAD rows. For each row: build the standard QA
    /// prompt, greedy-generate up to `maxNewTokens`, exact-match against
    /// any gold answer in `answers.text[]`.
    private static func runIndicGenBench(path: String, subtask: String,
                                          model: AnyModel, cfg: ModelConfig,
                                          tokenizer: HFTokenizer?,
                                          limit: Int?,
                                          maxNewTokens: Int,
                                          verbose: Bool) -> TaskResult {
        print("\n→ IndicGenBench (\(subtask))")
        if subtask != "xquad" {
            print("  ! subtask '\(subtask)' not implemented yet — only 'xquad' is wired.")
            print("    See docs/research/indic_evals.md §next-steps for cross-sum / xorqa / flores.")
            return TaskResult(total: 0, correct: 0)
        }
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("  ! data file not found: \(path)")
            print("    Pre-fetch with: tinygpt download-dataset google/IndicGenBench_xquad_in")
            return TaskResult(total: 0, correct: 0)
        }

        var seen = 0
        var correct = 0
        var skipped = 0
        do {
            _ = try RowReader.readRows(url: url, format: .jsonl) { row in
                if let cap = limit, seen >= cap { return false }
                guard let q = row["question"] as? String,
                      let ctx = row["context"] as? String else {
                    skipped += 1; return true
                }
                let golds = extractXQuADGolds(row: row)
                if golds.isEmpty { skipped += 1; return true }

                let prompt = "Context: \(ctx)\nQuestion: \(q)\nAnswer:"
                let out = greedyGenerate(model: model, cfg: cfg,
                                          tokenizer: tokenizer,
                                          prompt: prompt,
                                          maxNewTokens: maxNewTokens)
                seen += 1
                let pred = normalizeAnswer(out)
                let hit = golds.contains(where: { normalizeAnswer($0) == pred && !pred.isEmpty })
                if hit { correct += 1 }
                if verbose || seen % 10 == 0 {
                    let em = 100.0 * Double(correct) / Double(seen)
                    fputs(String(format: "  [%4d]  EM=%.2f%%  pred=\"%@\"\n",
                                  seen, em, String(pred.prefix(40))), stderr)
                }
                return true
            }
        } catch {
            print("  ! read error: \(error)")
            return TaskResult(total: seen, correct: correct)
        }
        let em = seen == 0 ? 0 : 100.0 * Double(correct) / Double(seen)
        print(String(format: "  result: EM=%.2f%% (%d / %d)\(skipped > 0 ? "  (\(skipped) malformed rows skipped)" : "")",
                     em, correct, seen))
        return TaskResult(total: seen, correct: correct)
    }

    /// Pull the gold answer strings out of an IndicXQuAD row. Two
    /// shapes are seen in the wild: SQuAD-style `answers: {text: [...]}`
    /// and the flat `answer: "..."` shape.
    private static func extractXQuADGolds(row: [String: Any]) -> [String] {
        if let answers = row["answers"] as? [String: Any],
           let txt = answers["text"] as? [Any] {
            return txt.compactMap { $0 as? String }
        }
        if let s = row["answer"] as? String, !s.isEmpty { return [s] }
        if let arr = row["answers"] as? [Any] {
            return arr.compactMap { $0 as? String }
        }
        return []
    }

    /// SQuAD-style answer normalization: lowercase, strip articles, strip
    /// punctuation, collapse whitespace. Same recipe the official SQuAD
    /// scoring script uses (and that IndicGenBench inherits).
    private static func normalizeAnswer(_ s: String) -> String {
        var t = s.lowercased()
        // Strip leading/trailing whitespace, including newlines that the
        // greedy decoder commonly emits.
        t = t.trimmingCharacters(in: .whitespacesAndNewlines)
        // Truncate at the first newline — generation past the first line
        // is almost always a hallucinated follow-up question.
        if let nl = t.firstIndex(of: "\n") { t = String(t[..<nl]) }
        // Strip punctuation.
        let punct: Set<Character> = [".", ",", "!", "?", ";", ":", "\"", "'", "(", ")", "[", "]", "{", "}"]
        t.removeAll { punct.contains($0) }
        // Collapse whitespace.
        let collapsed = t.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        // Strip English articles. (XQuAD answers in Indic scripts won't
        // hit this; harmless for them.)
        let parts = collapsed.split(separator: " ").filter {
            $0 != "a" && $0 != "an" && $0 != "the"
        }
        return parts.joined(separator: " ")
    }

    // MARK: - Generation + tokenization helpers

    private static func encode(text: String, tokenizer: HFTokenizer?) -> [Int] {
        if let tok = tokenizer {
            return (try? tok.encode(text)) ?? []
        }
        return [UInt8](text.utf8).map { Int($0) }
    }

    /// Greedy generation — mirrors `Score.greedyGenerate` (kept private
    /// there). Same simple O(n²) form: re-feed the full window each step.
    /// Acceptable for short answers (XQuAD: 32 tokens cap).
    private static func greedyGenerate(model: AnyModel, cfg: ModelConfig,
                                        tokenizer: HFTokenizer?,
                                        prompt: String,
                                        maxNewTokens: Int) -> String {
        let promptIds = encode(text: prompt, tokenizer: tokenizer).map { Int32($0) }
        // Left-truncate to ctx so a long context doesn't trip the kernel.
        let ctx = cfg.contextLength
        let seed: [Int32]
        if promptIds.count > ctx {
            seed = Array(promptIds[(promptIds.count - ctx)..<promptIds.count])
        } else {
            seed = promptIds
        }
        var idx = MLXArray(seed, [1, seed.count])
        var generated: [Int32] = []
        for _ in 0..<maxNewTokens {
            let T = idx.shape.last!
            let lo = max(0, T - ctx)
            let cond = idx[0..., lo..<T]
            let logits = model(cond)
            let last = logits[0..., logits.shape[1] - 1, 0...]
            let next = MLX.argMax(last, axis: -1).reshaped([1, 1])
            eval(next)
            let id = Int32(next.item(Int32.self))
            generated.append(id)
            idx = concatenated([idx, next.asType(idx.dtype)], axis: 1)
        }
        if let tok = tokenizer {
            return tok.decode(generated.map { Int($0) })
        }
        // Byte-level: print only valid bytes.
        var s = ""
        for id in generated {
            if let scalar = UnicodeScalar(Int(id)), id >= 9 {
                s.append(Character(scalar))
            }
        }
        return s
    }

    // MARK: - Usage

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt eval-indic --task <milu|indicgenbench|all> --model <path> [options]

        --task <name>             milu | indicgenbench | all  (required)
        --subtask <name>          indicgenbench subtask (default: xquad — only one wired)
        --model <path>            .tinygpt or HF model dir (required)
        --milu-data <path.jsonl>  MILU JSONL (required for task=milu|all)
        --indicgen-data <path>    IndicGenBench JSONL (required for task=indicgenbench|all)
        --limit N                 Cap rows per task (default: all)
        --max-new-tokens N        XQuAD answer-gen cap (default: 32)
        --verbose                 Per-row progress

        Pre-fetch datasets:
          tinygpt download-dataset ai4bharat/MILU
          tinygpt download-dataset google/IndicGenBench_xquad_in

        See docs/research/indic_evals.md for schema, scoring, baselines,
        and the published-paper context.
        """)
        exit(2)
    }
}

