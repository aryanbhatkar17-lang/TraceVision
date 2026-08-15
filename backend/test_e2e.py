import urllib.request
import json
import sys

def test_homepage_empty_state():
    print("\n=== 1. Testing Initial Empty Dashboard State (GET /) ===")
    req = urllib.request.Request("http://localhost:3000", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as res:
        html = res.read().decode("utf-8")
        print(f"HTTP Status: {res.status}")
        assert res.status == 200

        # Check required tactical empty state phrases
        assert "No CCTV Feed Active" in html, "Missing 'No CCTV Feed Active' dropzone"
        assert "Drag and drop or click" in html, "Missing upload CTA text"
        assert "Upload Footage" in html, "Missing 'Upload Footage' text"
        assert "0 matches" in html, "Missing '0 matches' in results panel"
        assert "No audit queries executed" in html, "Missing audit empty message"
        assert "0 EVENTS FLAGGED" in html, "Missing 0 events flagged in timeline"

        # Check that mock data is NOT present
        assert "Delivery person entering frame" not in html, "Residual mock data: Delivery person"
        assert "Red hatchback crossing junction" not in html, "Residual mock data: Red hatchback"
        assert "Individual lingering near garage" not in html, "Residual mock data: Individual lingering"
        assert "photo-1518709268805" not in html, "Residual mock image URL found"
        assert "photo-1506521781263" not in html, "Residual mock image URL found"

        print("  [PASS] Empty state on initial load verified with ZERO mock data!")

def test_analyze_endpoint():
    print("\n=== 2. Testing Analyze Pipeline & Strict Schema (POST /api/analyze) ===")
    boundary = "----SentinelBoundary12345"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="query"\r\n\r\n'
        f"Locate delivery person or green city bus\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="duration"\r\n\r\n'
        f"180\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:3000/api/analyze",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        raw_json = res.read().decode("utf-8")
        data = json.loads(raw_json)

        print(f"HTTP Status: {res.status}")
        print(f"Total Chunks: {data.get('total_chunks')}")
        print(f"Video Duration: {data.get('video_duration')}s")
        print(f"Matches count: {len(data.get('matches', []))}")

        assert "matches" in data, "Response missing 'matches' key"
        assert len(data["matches"]) > 0, "Response returned 0 matches"

        for idx, match in enumerate(data["matches"]):
            print(f"  Match #{idx+1}: [{match['start_time']} - {match['end_time']}] "
                  f"({match['start_seconds']}s - {match['end_seconds']}s) "
                  f"[{match['category']}]: {match['description']}")

            # Strict schema validation
            assert "start_time" in match, f"Match {idx} missing start_time"
            assert "end_time" in match, f"Match {idx} missing end_time"
            assert "start_seconds" in match, f"Match {idx} missing start_seconds"
            assert "end_seconds" in match, f"Match {idx} missing end_seconds"
            assert "category" in match, f"Match {idx} missing category"
            assert "description" in match, f"Match {idx} missing description"
            assert match["end_seconds"] >= match["start_seconds"], "Invalid timestamp range"
            assert "thumbnail" not in match, "Match should not contain thumbnail field"

        print("  [PASS] Strict JSON schema compliance and timestamp mapping verified!")

def test_upload_endpoint():
    print("\n=== 3. Testing Upload Endpoint (POST /api/upload) ===")
    boundary = "----SentinelBoundaryUpload"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="perimeter_cam4.mp4"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
        f"FAKE_VIDEO_STREAM_BYTES_FOR_UPLOAD_TEST\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:3000/api/upload",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(req) as res:
        assert res.status == 200
        data = json.loads(res.read().decode("utf-8"))
        print(f"HTTP Status: {res.status}")
        print(f"Uploaded video metadata: {data}")
        assert "video_id" in data or "original_filename" in data
        print("  [PASS] Upload endpoint verified!")

if __name__ == "__main__":
    try:
        test_homepage_empty_state()
        test_analyze_endpoint()
        test_upload_endpoint()
        print("\n=======================================================")
        print(">>> ALL SENTINEL VERIFICATION SUITES PASSED (100%) <<<")
        print("=======================================================\n")
    except Exception as e:
        print(f"\n[FAIL] Test encountered error: {e}", file=sys.stderr)
        sys.exit(1)
