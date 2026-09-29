// One-off script: lists everything on the network volume with sizes, so
// you can see what's using space without waiting on a slow check from me.
//
// Usage:
//   node list-volume.js
require("dotenv").config();
const { S3Client, ListObjectsV2Command } = require("@aws-sdk/client-s3");

const {
  RUNPOD_S3_ENDPOINT,
  RUNPOD_S3_ACCESS_KEY,
  RUNPOD_S3_SECRET_KEY,
  RUNPOD_VOLUME_ID,
} = process.env;

for (const [name, value] of Object.entries({
  RUNPOD_S3_ENDPOINT,
  RUNPOD_S3_ACCESS_KEY,
  RUNPOD_S3_SECRET_KEY,
  RUNPOD_VOLUME_ID,
})) {
  if (!value) {
    console.error(`Missing ${name} - fill in .env first (see .env.example).`);
    process.exit(1);
  }
}

const s3 = new S3Client({
  endpoint: RUNPOD_S3_ENDPOINT,
  region: "eur-no-1",
  credentials: {
    accessKeyId: RUNPOD_S3_ACCESS_KEY,
    secretAccessKey: RUNPOD_S3_SECRET_KEY,
  },
  forcePathStyle: true,
});

async function main() {
  let total = 0;
  let continuationToken = undefined;
  const rows = [];

  do {
    const resp = await s3.send(new ListObjectsV2Command({
      Bucket: RUNPOD_VOLUME_ID,
      ContinuationToken: continuationToken,
    }));
    for (const obj of resp.Contents || []) {
      total += obj.Size;
      rows.push(obj);
    }
    continuationToken = resp.IsTruncated ? resp.NextContinuationToken : undefined;
  } while (continuationToken);

  rows.sort((a, b) => b.Size - a.Size);
  for (const obj of rows) {
    console.log(`${(obj.Size / 1e9).toFixed(2)} GB  ${obj.Key}`);
  }
  console.log(`\nTotal: ${(total / 1e9).toFixed(2)} GB (volume is 90GB)`);
}

main().catch((err) => {
  console.error("List failed:", err);
  process.exit(1);
});
