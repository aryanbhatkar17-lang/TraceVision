import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  serverExternalPackages: ['fluent-ffmpeg', 'ffmpeg-static', 'sharp'],

  // Required headers for FFmpeg.wasm SharedArrayBuffer threading support.
  // Both COOP and COEP must be present or SharedArrayBuffer is unavailable
  // and FFmpeg.wasm falls back to a slower, single-threaded build.
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Cross-Origin-Opener-Policy',
            value: 'same-origin',
          },
          {
            key: 'Cross-Origin-Embedder-Policy',
            value: 'credentialless',
          },
        ],
      },
    ];
  },

  // Webpack override: suppress FFmpeg.wasm's internal dynamic-require warnings.
  // @ffmpeg/ffmpeg uses require(someVariable) internally for its WASM loader
  // path resolution. Webpack would normally emit a "Critical dependency: the
  // request of a CommonJS require() is an expression" error that crashes the
  // build. Setting exprContextCritical = false downgrades these from build
  // errors to silent warnings, allowing the bundle to complete correctly.
  webpack: (config: { module?: { exprContextCritical?: boolean } }) => {
    config.module = config.module ?? {};
    config.module.exprContextCritical = false;
    return config;
  },
};

export default nextConfig;