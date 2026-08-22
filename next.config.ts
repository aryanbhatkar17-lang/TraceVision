import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  serverExternalPackages: ['fluent-ffmpeg', 'ffmpeg-static', 'sharp',],
};

export default nextConfig;