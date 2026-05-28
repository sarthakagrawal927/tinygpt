import Foundation
import XCTest
@testable import TinyGPTIO

final class TinyGPTIOTests: XCTestCase {

    // MARK: - Helpers

    /// Build a synthetic file with two small tensors. Used by the round-trip tests.
    private func makeFixtureFile() -> TinyGPTFile {
        let entries: [TinyGPTHeader.TensorEntry] = [
            .init(name: "token_embedding.weight", shape: [4, 8]),
            .init(name: "output.weight", shape: [8, 4]),
        ]
        let header = TinyGPTHeader(
            config: .init(layers: 1, dModel: 8, ctx: 16, heads: 2, dMlp: 16, batchSize: 2, backend: "wasm"),
            manifest: entries,
            savedAt: "2026-05-28T00:00:00Z",
            finalLoss: .init(step: 100, train: 1.234, val: 1.456),
            sample: "hello world"
        )
        let tensors = entries.enumerated().map { (i, entry) -> TinyGPTTensor in
            // Distinct, easily-checkable byte patterns per tensor.
            let n = entry.elementCount
            let weight = Self.floats(repeatingPattern: Float(i + 1) * 0.5, count: n)
            let m = Self.floats(repeatingPattern: Float(i + 1) * 0.25, count: n)
            let v = Self.floats(repeatingPattern: Float(i + 1) * 0.125, count: n)
            return TinyGPTTensor(entry: entry, weight: weight, adamM: m, adamV: v)
        }
        return TinyGPTFile(header: header, step: 42, tensors: tensors)
    }

    private static func floats(repeatingPattern value: Float, count: Int) -> Data {
        let buf = [Float](repeating: value, count: count)
        return buf.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func tmpURL(_ name: String = "tinygpt-test.tinygpt") -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinygpt-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent(name)
    }

    // MARK: - Header decode

    func test_decodesMinimalHeaderJSON() throws {
        let json = """
        {
          "config": { "layers": 2, "dModel": 16, "ctx": 32 },
          "manifest": [
            { "name": "a.weight", "shape": [4, 4] }
          ]
        }
        """
        let header = try JSONDecoder().decode(TinyGPTHeader.self, from: Data(json.utf8))
        XCTAssertEqual(header.config.layers, 2)
        XCTAssertEqual(header.config.dModel, 16)
        XCTAssertEqual(header.config.ctx, 32)
        XCTAssertEqual(header.manifest.count, 1)
        XCTAssertEqual(header.manifest[0].name, "a.weight")
        XCTAssertEqual(header.manifest[0].elementCount, 16)
    }

    func test_ignoresUnknownTopLevelKeys() throws {
        // Browser ships extra fields (savedAt, finalLoss, lossHistory, gpuBytes, etc.)
        // — the reader must not choke on ones we don't model explicitly.
        let json = """
        {
          "config": { "layers": 1 },
          "manifest": [{ "name": "x", "shape": [2] }],
          "gpuBytes": 9999,
          "futureField": { "nested": [1, 2, 3] }
        }
        """
        XCTAssertNoThrow(
            try JSONDecoder().decode(TinyGPTHeader.self, from: Data(json.utf8))
        )
    }

    // MARK: - Round-trip

    func test_roundTripsByteIdenticallyWhenJSONKeysAreSorted() throws {
        let original = makeFixtureFile()
        let encoded = try TinyGPTFileWriter.encode(original)
        let decoded = try TinyGPTFileReader.decode(encoded, source: tmpURL())

        XCTAssertEqual(decoded.version, original.version)
        XCTAssertEqual(decoded.step, original.step)
        XCTAssertEqual(decoded.header, original.header)
        XCTAssertEqual(decoded.tensors.count, original.tensors.count)
        for (a, b) in zip(original.tensors, decoded.tensors) {
            XCTAssertEqual(a.entry, b.entry)
            XCTAssertEqual(a.weight, b.weight)
            XCTAssertEqual(a.adamM, b.adamM)
            XCTAssertEqual(a.adamV, b.adamV)
        }

        // Second encode reproduces the same bytes — the writer is deterministic.
        let reencoded = try TinyGPTFileWriter.encode(decoded)
        XCTAssertEqual(reencoded, encoded)
    }

    func test_writeThenReadFromDisk() throws {
        let url = tmpURL()
        let original = makeFixtureFile()
        try TinyGPTFileWriter.write(original, to: url)
        let loaded = try TinyGPTFileReader.read(url)
        XCTAssertEqual(loaded.header, original.header)
        XCTAssertEqual(loaded.step, original.step)
        XCTAssertEqual(loaded.tensors.count, original.tensors.count)
        for (a, b) in zip(original.tensors, loaded.tensors) {
            XCTAssertEqual(a.weightFloats, b.weightFloats)
            XCTAssertEqual(a.adamMFloats, b.adamMFloats)
            XCTAssertEqual(a.adamVFloats, b.adamVFloats)
        }
    }

    // MARK: - Error paths

    func test_rejectsBadMagic() {
        var data = Data([0x4e, 0x4f, 0x50, 0x45])  // "NOPE"
        data.append(Data(count: 8))
        XCTAssertThrowsError(try TinyGPTFileReader.decode(data, source: tmpURL()))
    }

    func test_rejectsUnsupportedVersion() {
        var data = Data()
        data.append(contentsOf: TinyGPTFormat.magic)
        var version: UInt32 = 99
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        var headerLen: UInt32 = 0
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        XCTAssertThrowsError(try TinyGPTFileReader.decode(data, source: tmpURL()))
    }

    func test_rejectsTruncatedBody() throws {
        // Build a valid file then chop the last few bytes off so the last tensor
        // is incomplete. The reader should fail with a `truncatedBody` error.
        let file = makeFixtureFile()
        var encoded = try TinyGPTFileWriter.encode(file)
        encoded.removeSubrange((encoded.count - 4)..<encoded.count)
        XCTAssertThrowsError(try TinyGPTFileReader.decode(encoded, source: tmpURL()))
    }

    func test_reportsMissingManifest() throws {
        // Header without `manifest` should produce the dedicated v1-detection error.
        var data = Data()
        data.append(contentsOf: TinyGPTFormat.magic)
        var version: UInt32 = 2
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        let json = #"{"config":{"layers":1}}"#
        let headerBytes = Data(json.utf8)
        var headerLen = UInt32(headerBytes.count)
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        data.append(headerBytes)
        data.append(Data(count: 4))  // step counter

        var sawMissingManifest = false
        do {
            _ = try TinyGPTFileReader.decode(data, source: tmpURL())
        } catch TinyGPTFileError.missingManifest {
            sawMissingManifest = true
        } catch {
            // Other error types are acceptable as long as the reader rejected the file.
        }
        XCTAssertTrue(sawMissingManifest)
    }

    // MARK: - fp16 inference layout

    /// Build a fp16 inference-layout file: weight buffer is contiguous, indexed
    /// by floatOffset. No AdamW state.
    private func makeFP16Fixture() -> TinyGPTFile {
        let entries: [TinyGPTHeader.TensorEntry] = [
            .init(name: "a", shape: [2, 2], floatOffset: 0),
            .init(name: "b", shape: [2], floatOffset: 4),
            .init(name: "c", shape: [3], floatOffset: 6),
        ]
        let header = TinyGPTHeader(
            config: .init(layers: 1),
            manifest: entries,
            weightDtype: "fp16",
            includesOptimizerState: false,
            stateByteLength: 9 * 2
        )
        // Build a contiguous fp16 buffer with distinct values per tensor.
        let halfA: [UInt16] = [0x3C00, 0x4000, 0x4200, 0x4400]  // 1.0, 2.0, 3.0, 4.0
        let halfB: [UInt16] = [0x4500, 0x4600]                    // 5.0, 6.0
        let halfC: [UInt16] = [0x4700, 0x4800, 0x4880]            // 7.0, 8.0, 9.0
        func bytes(_ halves: [UInt16]) -> Data {
            return halves.withUnsafeBufferPointer { ptr in
                Data(buffer: UnsafeBufferPointer(start: UnsafeRawPointer(ptr.baseAddress)!
                    .bindMemory(to: UInt8.self, capacity: halves.count * 2),
                    count: halves.count * 2))
            }
        }
        let tensors = [
            TinyGPTTensor(entry: entries[0], weight: bytes(halfA), dtype: .fp16),
            TinyGPTTensor(entry: entries[1], weight: bytes(halfB), dtype: .fp16),
            TinyGPTTensor(entry: entries[2], weight: bytes(halfC), dtype: .fp16),
        ]
        return TinyGPTFile(header: header, step: 1000, tensors: tensors)
    }

    func test_detectsFP16BodyLayout() throws {
        let f = makeFP16Fixture()
        XCTAssertEqual(f.header.bodyLayout, .inferenceFP16)
    }

    func test_roundTripsFP16InferenceLayout() throws {
        let original = makeFP16Fixture()
        let encoded = try TinyGPTFileWriter.encode(original)
        let decoded = try TinyGPTFileReader.decode(encoded, source: tmpURL())

        XCTAssertEqual(decoded.header, original.header)
        XCTAssertEqual(decoded.step, original.step)
        XCTAssertEqual(decoded.tensors.count, original.tensors.count)
        for (a, b) in zip(original.tensors, decoded.tensors) {
            XCTAssertEqual(a.entry, b.entry)
            XCTAssertEqual(a.weight, b.weight)
            XCTAssertEqual(b.dtype, .fp16)
            XCTAssertTrue(b.adamM.isEmpty)
            XCTAssertTrue(b.adamV.isEmpty)
        }
    }

    func test_fp16WeightExpandsToFloat32Correctly() throws {
        let original = makeFP16Fixture()
        let encoded = try TinyGPTFileWriter.encode(original)
        let decoded = try TinyGPTFileReader.decode(encoded, source: tmpURL())
        let tensorA = decoded.tensors[0]
        XCTAssertEqual(tensorA.weightFP16AsFloat32(), [1.0, 2.0, 3.0, 4.0])
        let tensorB = decoded.tensors[1]
        XCTAssertEqual(tensorB.weightFP16AsFloat32(), [5.0, 6.0])
        let tensorC = decoded.tensors[2]
        XCTAssertEqual(tensorC.weightFP16AsFloat32(), [7.0, 8.0, 9.0])
    }

    // MARK: - TensorEntry geometry

    func test_tensorEntryByteLengthIsShapeProductTimesFour() {
        let e = TinyGPTHeader.TensorEntry(name: "x", shape: [3, 4, 5])
        XCTAssertEqual(e.elementCount, 60)
        XCTAssertEqual(e.byteLength, 240)
    }
}
