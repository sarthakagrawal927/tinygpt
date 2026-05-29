import Foundation
import MLX
import MLXNN

/// Custom-function-based gradient checkpointing.
///
/// MLX-Swift (as of v0.25) does NOT expose a first-class `mlx.checkpoint`
/// primitive. The mechanism we use instead: wrap each TransformerBlock's
/// forward in an MLX `CustomFunction` whose VJP re-runs the same forward
/// inside `MLX.vjp(...)` to compute gradients. This produces the
/// canonical activation-checkpointing memory profile — the block's
/// intermediate activations are dropped after forward and recomputed
/// lazily at backward time — at the cost of one extra forward per
/// checkpointed block (~30% step-time overhead).
///
/// The block's parameters are threaded through the CustomFunction as
/// declared inputs so MLX's autodiff propagates gradients back to them
/// correctly. This mirrors how `valueAndGrad(model:)` injects the
/// model's trainable parameters as primal inputs into the gradient
/// transform — the parameters arrive as tracers at trace time, the
/// wrapper splices them into the block via `block.update(parameters:)`
/// before running the forward, and the autograd sees a normal data-flow
/// graph for the recomputation.
///
/// Subtleties:
///   * The "raw" forward (passed in as `forward:`) MUST be the version
///     that does NOT recurse into another checkpoint wrapper. Otherwise
///     the VJP would loop infinitely. TransformerBlock + TransformerBlockHF
///     each expose a private `rawForward(_:)` for this.
///   * `block.update(parameters:)` mutates the block's @ModuleInfo slots.
///     Since both call sites (Forward and VJP) re-slot the tracers each
///     invocation, the mutation is local to the trace and doesn't leak.
public enum GradCheckpoint {

    /// Run `forward` on `block` with activation checkpointing applied.
    ///
    /// At backward time, the same forward is re-executed inside
    /// `MLX.vjp(_:primals:cotangents:)` to compute gradients w.r.t. the
    /// input `x` AND every flattened parameter of the block. Returns
    /// the block's output.
    ///
    /// - Parameters:
    ///   - block: the Module whose parameters take part in the forward.
    ///   - x: input tensor.
    ///   - forward: closure that, given a freshly updated block and an
    ///     input tensor, produces the block's output. MUST be the raw
    ///     forward (no further checkpoint wrapping) to avoid recursion
    ///     in the VJP.
    public static func wrap<M: Module>(
        block: M,
        x: MLXArray,
        forward: @escaping (M, MLXArray) -> MLXArray
    ) -> MLXArray {
        // Capture the block's current parameter layout. We need the
        // flat keys at trace time so the Forward / VJP closures can
        // unflatten incoming tracers back into the structured update.
        let trainable = block.trainableParameters().flattened()
        let flatKeys = trainable.map { $0.0 }
        let flatParams = trainable.map { $0.1 }

        // Inner function that the CustomFunction's VJP will differentiate
        // through. Splices the flattened param tracers back into `block`
        // and runs the user-supplied raw forward.
        let runForward: ([MLXArray]) -> [MLXArray] = { inputs in
            let xt = inputs[0]
            let paramArrays = Array(inputs.dropFirst())
            let tuples = zip(flatKeys, paramArrays).map { ($0.0, $0.1) }
            let nested = NestedDictionary<String, MLXArray>.unflattened(tuples)
            // update() is a no-throw discardableResult that re-slots the
            // tracers into the @ModuleInfo properties. The raw forward
            // doesn't re-enter the checkpoint wrapper.
            block.update(parameters: nested)
            return [forward(block, xt)]
        }

        let cf = CustomFunction {
            Forward { inputs in runForward(inputs) }
            VJP { primals, cotangents in
                // Re-execute the forward and ask MLX to compute the
                // vector-Jacobian product. The returned gradients are
                // ordered to match `primals` — [dx, dp_0, dp_1, ...].
                // Because vjp runs the forward AGAIN inside this closure,
                // the block's intermediate activations are NOT retained
                // across the outer backward — that's the memory win.
                let (_, grads) = MLX.vjp(runForward,
                                          primals: primals,
                                          cotangents: cotangents)
                return grads
            }
        }

        let out = cf([x] + flatParams)
        return out[0]
    }
}
