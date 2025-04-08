// app/page.tsx
"use client";

import { useState } from "react";

const MATRIX_SIZE = 4;
const BUFFER_SIZE = MATRIX_SIZE * MATRIX_SIZE * 4; // Float32: 4 bytes each

// WGSL shader for matrix multiplication
const shaderCode = /* wgsl */ `
struct Matrix {
  numbers: array<f32>,
};

@group(0) @binding(0) var<storage, read> matrixA: Matrix;
@group(0) @binding(1) var<storage, read> matrixB: Matrix;
@group(0) @binding(2) var<storage, write> matrixOut: Matrix;

const MATRIX_SIZE: u32 = ${MATRIX_SIZE};

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= MATRIX_SIZE || id.y >= MATRIX_SIZE) {
    return;
  }
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < MATRIX_SIZE; i = i + 1u) {
    let aIndex = id.y * MATRIX_SIZE + i;
    let bIndex = i * MATRIX_SIZE + id.x;
    sum = sum + matrixA.numbers[aIndex] * matrixB.numbers[bIndex];
  }
  let index = id.y * MATRIX_SIZE + id.x;
  matrixOut.numbers[index] = sum;
}
`;

const createDefaultMatrix = () => {
  // Create a default 4x4 matrix filled with 0's.
  return Array.from({ length: MATRIX_SIZE }, () =>
    Array.from({ length: MATRIX_SIZE }, () => 0)
  );
};

export default function HomePage() {
  const [matrixA, setMatrixA] = useState<number[][]>(() => {
    // Identity matrix example
    return createDefaultMatrix().map((row, i) =>
      row.map((_, j) => (i === j ? 1 : 0))
    );
  });
  const [matrixB, setMatrixB] = useState<number[][]>(() => {
    // Example matrix
    return createDefaultMatrix().map((row, i) =>
      row.map((_, j) => i * MATRIX_SIZE + j + 1)
    );
  });
  const [resultMatrix, setResultMatrix] = useState<number[][] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Helper function to update a matrix cell
  const updateMatrix = (
    matrix: number[][],
    setMatrix: (m: number[][]) => void,
    row: number,
    col: number,
    value: string
  ) => {
    const num = parseFloat(value);
    const newMatrix = [...matrix];
    newMatrix[row] = [...matrix[row]];
    newMatrix[row][col] = isNaN(num) ? 0 : num;
    setMatrix(newMatrix);
  };

  // WebGPU multiplication logic
  const multiplyMatrices = async () => {
    setError(null);
    setLoading(true);
    try {
      if (!navigator.gpu) {
        throw new Error("WebGPU is not supported on this browser.");
      }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("Failed to get GPU adapter.");
      const device = await adapter.requestDevice();

      // Flatten matrix arrays to Float32Array
      const flatten = (matrix: number[][]) =>
        new Float32Array(matrix.flat());
      const aData = flatten(matrixA);
      const bData = flatten(matrixB);

      // Create GPU buffers
      const aBuffer = device.createBuffer({
        size: BUFFER_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(aBuffer, 0, aData.buffer, aData.byteOffset, aData.byteLength);

      const bBuffer = device.createBuffer({
        size: BUFFER_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(bBuffer, 0, bData.buffer, bData.byteOffset, bData.byteLength);

      const resultBuffer = device.createBuffer({
        size: BUFFER_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      // Create shader module and compute pipeline
      const shaderModule = device.createShaderModule({
        code: shaderCode,
      });
      const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
          module: shaderModule,
          entryPoint: "main",
        },
      });

      // Create bind group to pass buffers to shader
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: aBuffer } },
          { binding: 1, resource: { buffer: bBuffer } },
          { binding: 2, resource: { buffer: resultBuffer } },
        ],
      });

      // Encode the commands
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      // Dispatch with enough workgroups for a 4x4 grid. (Workgroup size is 8x8.)
      passEncoder.dispatchWorkgroups(1, 1);
      passEncoder.end();

      device.queue.submit([commandEncoder.finish()]);

      // Copy result buffer to a mappable buffer
      const readBuffer = device.createBuffer({
        size: BUFFER_SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const copyEncoder = device.createCommandEncoder();
      copyEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, BUFFER_SIZE);
      device.queue.submit([copyEncoder.finish()]);

      await readBuffer.mapAsync(GPUMapMode.READ);
      const resultArrayBuffer = readBuffer.getMappedRange();
      const resultData = new Float32Array(resultArrayBuffer.slice(0));
      readBuffer.unmap();

      // Convert flat result to 2D matrix
      const newResultMatrix: number[][] = [];
      for (let i = 0; i < MATRIX_SIZE; i++) {
        newResultMatrix.push(
          Array.from(resultData.slice(i * MATRIX_SIZE, i * MATRIX_SIZE + MATRIX_SIZE))
        );
      }
      setResultMatrix(newResultMatrix);
    } catch (e: any) {
      setError(e.message);
    }
    setLoading(false);
  };

  return (
    <main className="min-h-screen bg-black p-8">
      <h1 className="text-4xl font-bold text-center mb-8 text-purple-400">
        WebGPU Matrix Multiplication
      </h1>
      {error && (
        <div className="bg-red-900 text-red-200 p-4 rounded mb-4 text-center">
          {error}
        </div>
      )}
      <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
        {/* Matrix A Input */}
        <div className="bg-gray-900 p-4 rounded shadow border border-gray-800">
          <h2 className="text-2xl font-semibold mb-4 text-center text-purple-300">Matrix A</h2>
          <div className="grid grid-cols-4 gap-2">
            {matrixA.map((row, rowIndex) =>
              row.map((value, colIndex) => (
                <input
                  key={`a-${rowIndex}-${colIndex}`}
                  type="number"
                  value={value}
                  onChange={(e) =>
                    updateMatrix(matrixA, setMatrixA, rowIndex, colIndex, e.target.value)
                  }
                  className="p-2 border border-gray-800 rounded text-center bg-gray-800 text-purple-200 focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              ))
            )}
          </div>
        </div>

        {/* Matrix B Input */}
        <div className="bg-gray-900 p-4 rounded shadow border border-gray-800">
          <h2 className="text-2xl font-semibold mb-4 text-center text-purple-300">Matrix B</h2>
          <div className="grid grid-cols-4 gap-2">
            {matrixB.map((row, rowIndex) =>
              row.map((value, colIndex) => (
                <input
                  key={`b-${rowIndex}-${colIndex}`}
                  type="number"
                  value={value}
                  onChange={(e) =>
                    updateMatrix(matrixB, setMatrixB, rowIndex, colIndex, e.target.value)
                  }
                  className="p-2 border border-gray-800 rounded text-center bg-gray-800 text-purple-200 focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              ))
            )}
          </div>
        </div>

        {/* Result Display */}
        <div className="bg-gray-900 p-4 rounded shadow border border-gray-800 flex flex-col items-center">
          <h2 className="text-2xl font-semibold mb-4 text-purple-300">Result</h2>
          {loading ? (
            <div className="text-lg text-purple-300">Multiplying...</div>
          ) : resultMatrix ? (
            <div className="grid grid-cols-4 gap-2">
              {resultMatrix.map((row, rowIndex) =>
                row.map((value, colIndex) => (
                  <div
                    key={`res-${rowIndex}-${colIndex}`}
                    className="p-2 border border-gray-800 rounded text-center bg-gray-800 text-purple-200"
                  >
                    {value.toFixed(2)}
                  </div>
                ))
              )}
            </div>
          ) : (
            <div className="text-lg text-gray-500">No result yet</div>
          )}
        </div>
      </div>
      <div className="text-center mt-8">
        <button
          onClick={multiplyMatrices}
          className="px-6 py-3 bg-purple-700 hover:bg-purple-800 transition text-white font-semibold rounded disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-purple-500"
          disabled={loading}
        >
          Multiply Matrices
        </button>
      </div>
    </main>
  );
}