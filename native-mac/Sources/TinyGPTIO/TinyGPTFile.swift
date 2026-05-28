import Foundation

/// Reader/writer for the `.tinygpt` binary format produced by the browser
/// playground and consumed by `python_ref/load_tinygpt.py`.
///
/// File layout (little-endian throughout):
///
/// ```
///   offset  bytes  field
///   ------  -----  ------------------------------------------
///   0       4      magic = "TGPT"
///   4       4      version (u32)             — 1 or 2
///   8       4      header_len (u32)
///   12      H      JSON header (UTF-8, length = header_len)
///   12+H    4      step counter (i32)
///   ...            per-tensor [weight, m, v] triplets,
///                   each fp32 row-major, `elementCount * 4` bytes
/// ```
///
/// Only version 2 is fully supported — v1 predates the `manifest` field and
/// can't be decoded without an external schema.
public enum TinyGPTFormat {
    public static let magic: [UInt8] = Array("TGPT".utf8)
    public static let supportedVersions: Set<UInt32> = [1, 2]
    public static let currentVersion: UInt32 = 2
}

/// In-memory representation of a fully decoded `.tinygpt` file.
public struct TinyGPTFile: Sendable {
    public var version: UInt32
    public var header: TinyGPTHeader
    public var step: Int32
    /// One `TinyGPTTensor` per manifest entry, in the same order.
    public var tensors: [TinyGPTTensor]

    public init(
        version: UInt32 = TinyGPTFormat.currentVersion,
        header: TinyGPTHeader,
        step: Int32 = 0,
        tensors: [TinyGPTTensor]
    ) {
        self.version = version
        self.header = header
        self.step = step
        self.tensors = tensors
    }
}

/// A single tensor's storage. The training-resumable layout populates `weight`,
/// `adamM`, `adamV` (all fp32). The inference fp16 layout populates only
/// `weight`, holding the raw fp16 bytes; `adamM` and `adamV` are empty.
public struct TinyGPTTensor: Sendable {
    public var entry: TinyGPTHeader.TensorEntry
    public var weight: Data
    public var adamM: Data
    public var adamV: Data
    /// `.fp32` for training-resumable, `.fp16` for inference distribution.
    public var dtype: TinyGPTDtype

    public init(
        entry: TinyGPTHeader.TensorEntry,
        weight: Data,
        adamM: Data = Data(),
        adamV: Data = Data(),
        dtype: TinyGPTDtype = .fp32
    ) {
        self.entry = entry
        self.weight = weight
        self.adamM = adamM
        self.adamV = adamV
        self.dtype = dtype
    }

    /// Convenience: return the weight buffer reinterpreted as `[Float32]`. Copies.
    /// Only valid for fp32 tensors; for fp16, use `weightFP16AsFloat32()` to
    /// expand to fp32 in-host.
    public var weightFloats: [Float32] { Self.toFloats(weight) }
    public var adamMFloats: [Float32] { Self.toFloats(adamM) }
    public var adamVFloats: [Float32] { Self.toFloats(adamV) }

    /// Decode a fp16 weight buffer into a fresh `[Float32]`. Useful for the
    /// inference layout where the on-disk storage is half-precision.
    public func weightFP16AsFloat32() -> [Float32] {
        precondition(dtype == .fp16, "weight is not fp16")
        let count = weight.count / 2
        var out = [Float32](repeating: 0, count: count)
        weight.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let halves = raw.bindMemory(to: UInt16.self)
            for i in 0..<count {
                out[i] = Float32(Float16(bitPattern: halves[i]))
            }
        }
        return out
    }

    static func toFloats(_ data: Data) -> [Float32] {
        let count = data.count / MemoryLayout<Float32>.size
        var out = [Float32](repeating: 0, count: count)
        _ = out.withUnsafeMutableBytes { dest in
            data.copyBytes(to: dest)
        }
        return out
    }
}

public enum TinyGPTDtype: Sendable, Equatable {
    case fp32
    case fp16
}

/// All errors thrown by the reader/writer. Each one names the file it was
/// trying to read so the CLI can surface the path in the message.
public enum TinyGPTFileError: Error, CustomStringConvertible, Sendable {
    case tooSmall(URL)
    case badMagic(URL, got: [UInt8])
    case unsupportedVersion(URL, got: UInt32)
    case truncatedHeader(URL, expected: Int, got: Int)
    case headerNotUTF8(URL)
    case headerNotJSON(URL, underlying: Error)
    case missingManifest(URL)
    case truncatedBody(URL, tensor: String, expected: Int, got: Int)

    public var description: String {
        switch self {
        case .tooSmall(let url):
            return "\(url.path): too small to be a .tinygpt file (< 12-byte prefix)"
        case .badMagic(let url, let got):
            let s = String(bytes: got, encoding: .ascii) ?? "<non-ascii>"
            return "\(url.path): bad magic \(s.debugDescription) (expected 'TGPT')"
        case .unsupportedVersion(let url, let got):
            return "\(url.path): unsupported file format version \(got) (supported: 1, 2)"
        case .truncatedHeader(let url, let expected, let got):
            return "\(url.path): header truncated — expected \(expected) bytes, got \(got)"
        case .headerNotUTF8(let url):
            return "\(url.path): header is not valid UTF-8"
        case .headerNotJSON(let url, let underlying):
            return "\(url.path): header is not valid JSON — \(underlying)"
        case .missingManifest(let url):
            return "\(url.path): no `manifest` array in header (v1 file? re-export from the browser to get v2)"
        case .truncatedBody(let url, let tensor, let expected, let got):
            return "\(url.path): tensor body truncated for \(tensor) — expected \(expected) bytes, got \(got)"
        }
    }
}

public enum TinyGPTFileReader {
    /// Read a `.tinygpt` file from disk. Reads the whole file into memory —
    /// fine for the model sizes we ship (max ~100 MB), and avoids the
    /// stream-cursor bookkeeping a chunked reader would need.
    public static func read(_ url: URL) throws -> TinyGPTFile {
        let data = try Data(contentsOf: url)
        return try decode(data, source: url)
    }

    /// Decode from an in-memory buffer. `source` is only used for error messages.
    public static func decode(_ data: Data, source: URL) throws -> TinyGPTFile {
        guard data.count >= 12 else {
            throw TinyGPTFileError.tooSmall(source)
        }
        let magicBytes = Array(data[0..<4])
        guard magicBytes == TinyGPTFormat.magic else {
            throw TinyGPTFileError.badMagic(source, got: magicBytes)
        }
        let version = readU32(data, at: 4)
        guard TinyGPTFormat.supportedVersions.contains(version) else {
            throw TinyGPTFileError.unsupportedVersion(source, got: version)
        }
        let headerLen = Int(readU32(data, at: 8))
        guard data.count >= 12 + headerLen + 4 else {
            throw TinyGPTFileError.truncatedHeader(
                source, expected: 12 + headerLen + 4, got: data.count
            )
        }
        let headerData = data.subdata(in: 12..<(12 + headerLen))
        guard String(data: headerData, encoding: .utf8) != nil else {
            throw TinyGPTFileError.headerNotUTF8(source)
        }
        let header: TinyGPTHeader
        do {
            header = try JSONDecoder().decode(TinyGPTHeader.self, from: headerData)
        } catch {
            // Detect the specific "missing manifest" case for a clearer error.
            if let obj = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any],
               obj["manifest"] == nil {
                throw TinyGPTFileError.missingManifest(source)
            }
            throw TinyGPTFileError.headerNotJSON(source, underlying: error)
        }

        // Past the header: int32 step counter, then either per-tensor triplets
        // (training layout) or a single contiguous weight buffer (inference layout).
        var cursor = 12 + headerLen
        let step = Int32(bitPattern: readU32(data, at: cursor))
        cursor += 4

        var tensors: [TinyGPTTensor] = []
        tensors.reserveCapacity(header.manifest.count)

        switch header.bodyLayout {
        case .trainingFP32:
            for entry in header.manifest {
                let need = entry.byteLengthFP32
                let triple = 3 * need
                guard data.count >= cursor + triple else {
                    throw TinyGPTFileError.truncatedBody(
                        source,
                        tensor: entry.name,
                        expected: cursor + triple,
                        got: data.count
                    )
                }
                let w = data.subdata(in: cursor..<(cursor + need)); cursor += need
                let m = data.subdata(in: cursor..<(cursor + need)); cursor += need
                let v = data.subdata(in: cursor..<(cursor + need)); cursor += need
                tensors.append(TinyGPTTensor(entry: entry, weight: w, adamM: m, adamV: v, dtype: .fp32))
            }

        case .inferenceFP16:
            // The body is one big fp16 buffer; each manifest entry slices into
            // it via `floatOffset`. Total bytes = 2 × Σ elementCount.
            let totalFloats = header.manifest.reduce(0) { $0 + $1.elementCount }
            let totalBytes = totalFloats * 2
            guard data.count >= cursor + totalBytes else {
                throw TinyGPTFileError.truncatedBody(
                    source,
                    tensor: "<fp16 weight buffer>",
                    expected: cursor + totalBytes,
                    got: data.count
                )
            }
            let bodyEnd = cursor + totalBytes
            let body = data.subdata(in: cursor..<bodyEnd)
            for entry in header.manifest {
                let offsetFloats = entry.floatOffset ?? 0
                let byteOffset = offsetFloats * 2
                let byteEnd = byteOffset + entry.byteLengthFP16
                guard byteEnd <= body.count else {
                    throw TinyGPTFileError.truncatedBody(
                        source,
                        tensor: entry.name,
                        expected: byteEnd,
                        got: body.count
                    )
                }
                let w = body.subdata(in: byteOffset..<byteEnd)
                tensors.append(TinyGPTTensor(entry: entry, weight: w, dtype: .fp16))
            }
            cursor = bodyEnd
        }

        return TinyGPTFile(version: version, header: header, step: step, tensors: tensors)
    }

    private static func readU32(_ data: Data, at offset: Int) -> UInt32 {
        let slice = data[offset..<(offset + 4)]
        return slice.withUnsafeBytes { raw in
            raw.loadUnaligned(as: UInt32.self).littleEndian
        }
    }
}

public enum TinyGPTFileWriter {
    /// Serialise a `TinyGPTFile` to the on-disk byte layout. The output is
    /// bit-identical for the same inputs across runs (deterministic JSON key
    /// order via `sortedKeys`, no embedded timestamps unless the caller put
    /// them in the header).
    public static func encode(_ file: TinyGPTFile) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let headerData = try encoder.encode(file.header)

        var out = Data()
        out.reserveCapacity(12 + headerData.count + 4 + file.tensors.reduce(0) { $0 + 3 * $1.entry.byteLength })

        // Magic + version + header_len
        out.append(contentsOf: TinyGPTFormat.magic)
        appendU32(&out, file.version)
        appendU32(&out, UInt32(headerData.count))
        out.append(headerData)
        // step counter
        appendU32(&out, UInt32(bitPattern: file.step))
        // tensor bodies — layout depends on header.bodyLayout
        switch file.header.bodyLayout {
        case .trainingFP32:
            for tensor in file.tensors {
                precondition(
                    tensor.weight.count == tensor.entry.byteLengthFP32 &&
                    tensor.adamM.count == tensor.entry.byteLengthFP32 &&
                    tensor.adamV.count == tensor.entry.byteLengthFP32,
                    "tensor \(tensor.entry.name) byte counts don't match manifest shape (fp32)"
                )
                out.append(tensor.weight)
                out.append(tensor.adamM)
                out.append(tensor.adamV)
            }
        case .inferenceFP16:
            // Layout is a contiguous fp16 buffer indexed by `floatOffset`. The
            // writer reassembles it by sorting tensors by floatOffset and
            // concatenating their fp16 weight bytes. Tensors without an offset
            // fall back to manifest order.
            let sorted = file.tensors.sorted { (a, b) in
                (a.entry.floatOffset ?? 0) < (b.entry.floatOffset ?? 0)
            }
            for tensor in sorted {
                precondition(
                    tensor.weight.count == tensor.entry.byteLengthFP16,
                    "tensor \(tensor.entry.name) byte count doesn't match manifest shape (fp16)"
                )
                out.append(tensor.weight)
            }
        }
        return out
    }

    /// Write a `TinyGPTFile` to disk. Atomic write — the destination either
    /// reflects the new file in full or is unchanged.
    public static func write(_ file: TinyGPTFile, to url: URL) throws {
        let data = try encode(file)
        try data.write(to: url, options: .atomic)
    }

    private static func appendU32(_ data: inout Data, _ value: UInt32) {
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { raw in
            data.append(contentsOf: raw)
        }
    }
}
