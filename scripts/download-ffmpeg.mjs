import fs from 'fs';
import path from 'path';
import https from 'https';

const ASSETS = [
  {
    url: 'https://unpkg.com/@ffmpeg/ffmpeg@0.12.10/dist/umd/ffmpeg.js',
    filename: 'ffmpeg.js'
  },
  {
    url: 'https://unpkg.com/@ffmpeg/util@0.12.2/dist/umd/index.js',
    filename: 'util.js'
  },
  {
    url: 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.js',
    filename: 'ffmpeg-core.js'
  },
  {
    url: 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.wasm',
    filename: 'ffmpeg-core.wasm'
  }
];

const TARGET_DIR = path.join(process.cwd(), 'public', 'ffmpeg');

if (!fs.existsSync(TARGET_DIR)) {
  fs.mkdirSync(TARGET_DIR, { recursive: true });
}

const downloadFile = (url, dest) => {
  return new Promise((resolve, reject) => {
    console.log(`Downloading ${path.basename(dest)}...`);
    const file = fs.createWriteStream(dest);
    
    // We need to handle redirects since unpkg might redirect
    const request = (currentUrl) => {
      https.get(currentUrl, (response) => {
        if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
          // Follow redirect
          request(response.headers.location.startsWith('http') 
            ? response.headers.location 
            : new URL(response.headers.location, currentUrl).href);
          return;
        }
        
        if (response.statusCode !== 200) {
          fs.unlink(dest, () => {}); // Delete the file async
          return reject(new Error(`Failed to get '${currentUrl}' (${response.statusCode})`));
        }
        
        response.pipe(file);
        
        file.on('finish', () => {
          file.close();
          resolve();
        });
      }).on('error', (err) => {
        fs.unlink(dest, () => {}); // Delete the file async
        reject(err);
      });
    };
    
    request(url);
  });
};

async function main() {
  console.log(`Downloading FFmpeg assets to ${TARGET_DIR} ...`);
  try {
    for (const asset of ASSETS) {
      const dest = path.join(TARGET_DIR, asset.filename);
      await downloadFile(asset.url, dest);
    }
    console.log('\nSuccess! All FFmpeg assets downloaded to public/ffmpeg/');
  } catch (error) {
    console.error('\nError downloading assets:', error);
    process.exit(1);
  }
}

main();
