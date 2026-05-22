// wgsl.d.ts — let TypeScript import .wgsl shader source as a string.
// Vite resolves the `?raw` suffix to the file's text content at build time.
declare module "*.wgsl?raw" {
  const source: string;
  export default source;
}
