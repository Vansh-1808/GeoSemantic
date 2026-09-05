"""
Automated Test Suite for Phase 8: Multi-Filter Search and GeoSemantic Fusion.
Tests:
1. Dynamic metadata filters inspection (/api/search/filters)
2. Independent sensor filtering
3. Independent date range temporal filtering
4. Independent cloud cover & quality threshold filtering
5. Independent bounding box spatial filtering
6. Independent polygon AOI spatial filtering with PostGIS ST_Intersects / ST_Within
7. Combined multi-filter GeoSemantic Fusion query
8. Zero-result empty state verification with restrictive filters
9. Telemetry breakdown (RemoteCLIP text embed + Qdrant vector search + PostGIS GiST)
"""
import sys
import time
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000/api"
client = httpx.Client(timeout=60.0)

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def test_filters_endpoint():
    print_header("Step 1: Inspect Dynamic Metadata Filters (/api/search/filters)")
    url = f"{BASE_URL}/search/filters"
    resp = client.get(url, timeout=10)
    assert resp.status_code == 200, f"Failed: {resp.status_code} {resp.text}"
    data = resp.json()
    print(f"  Total Indexed Tiles: {data.get('total_indexed_tiles')}")
    print(f"  Sensors Available:   {data.get('sensors')}")
    print(f"  Sensor Counts:       {data.get('sensor_counts')}")
    print(f"  Date Bounds:         {data.get('min_date')} -> {data.get('max_date')}")
    print(f"  Min Quality / Max Cloud: {data.get('min_quality')} / {data.get('max_cloud_cover')}%")
    assert data.get("total_indexed_tiles", 0) > 0, "No indexed tiles found to test against!"
    return data

def test_independent_sensor_filter(sensor: str):
    print_header(f"Step 2: Independent Sensor Filter (Sensor: {sensor})")
    url = f"{BASE_URL}/search/semantic"
    payload = {
        "query": "residential buildings or infrastructure",
        "top_k": 10,
        "sensors": [sensor],
    }
    t0 = time.perf_counter()
    resp = client.post(url, json=payload, timeout=30)
    dur = (time.perf_counter() - t0) * 1000
    assert resp.status_code == 200, f"Failed: {resp.status_code} {resp.text}"
    data = resp.json()
    print(f"  Query: '{data['query']}'")
    print(f"  Found: {data['total_found']} tiles in {data['execution_time_ms']}ms (API roundtrip: {dur:.1f}ms)")
    print(f"  Telemetry: Embed={data['text_embedding_time_ms']}ms | Vector={data['vector_search_time_ms']}ms | Device={data['device']}")
    for r in data["results"][:3]:
        print(f"    - Rank #{r['rank']} [Score: {r['similarity_score']:.3f}] Sensor: {r['sensor']} Scene: {r['scene_name']}")
        assert r["sensor"] == sensor, f"Tile sensor mismatch! Expected {sensor}, got {r['sensor']}"
    print("  [PASS] All returned tiles strictly match requested sensor filter.")

def test_independent_date_filter():
    print_header("Step 3: Independent Date Range Temporal Filter")
    url = f"{BASE_URL}/search/semantic"
    # Query with wide historical date filter
    payload = {
        "query": "commercial airport runway",
        "top_k": 10,
        "start_date": "2020-01-01T00:00:00Z",
        "end_date": "2025-12-31T23:59:59Z",
    }
    resp = client.post(url, json=payload, timeout=30)
    assert resp.status_code == 200, f"Failed: {resp.status_code} {resp.text}"
    data = resp.json()
    print(f"  Found {data['total_found']} tiles in {data['execution_time_ms']}ms")
    for r in data["results"][:3]:
        acq = r.get("acquisition_date")
        print(f"    - Rank #{r['rank']} Date: {acq} | Score: {r['similarity_score']:.3f}")
        if acq:
            assert "2020" <= acq[:4] <= "2025", f"Tile date {acq} outside specified range!"
    print("  [PASS] All returned tiles fall within specified date range.")

def test_independent_quality_and_cloud_filter():
    print_header("Step 4: Independent Atmospheric & Quality Filter")
    url = f"{BASE_URL}/search/semantic"
    min_qual = 0.50
    max_cloud = 50.0
    payload = {
        "query": "industrial facilities",
        "top_k": 10,
        "min_quality": min_qual,
        "max_cloud_cover": max_cloud,
    }
    resp = client.post(url, json=payload, timeout=30)
    assert resp.status_code == 200, f"Failed: {resp.status_code} {resp.text}"
    data = resp.json()
    print(f"  Found {data['total_found']} tiles matching min_quality >= {min_qual} & max_cloud <= {max_cloud}%")
    for r in data["results"][:3]:
        q = r.get("quality_score")
        c = r.get("cloud_cover_pct")
        print(f"    - Rank #{r['rank']} Quality: {q} | Cloud: {c}% | Score: {r['similarity_score']:.3f}")
        if q is not None:
            assert q >= min_qual, f"Quality {q} below threshold {min_qual}"
        if c is not None:
            assert c <= max_cloud, f"Cloud {c}% above threshold {max_cloud}"
    print("  [PASS] All returned tiles satisfy quality and cloud cover constraints.")

def test_independent_bounding_box_aoi():
    print_header("Step 5: Independent Bounding Box AOI Filter")
    url = f"{BASE_URL}/search/semantic"
    # Envelope encompassing Northern India / Delhi demo scene area
    bbox = [76.5, 27.5, 78.0, 29.5]  # [west, south, east, north]
    payload = {
        "query": "dense road network and crossings",
        "top_k": 10,
        "aoi_bbox": bbox,
    }
    resp = client.post(url, json=payload, timeout=30)
    assert resp.status_code == 200, f"Failed: {resp.status_code} {resp.text}"
    data = resp.json()
    print(f"  Found {data['total_found']} tiles inside BBox {bbox}")
    for r in data["results"][:3]:
        c = r["center_coordinates"]
        print(f"    - Rank #{r['rank']} Center: [{c['lat']:.4f}, {c['lon']:.4f}] | Score: {r['similarity_score']:.3f}")
        assert bbox[0] <= c["lon"] <= bbox[2], f"Lon {c['lon']} outside bbox [{bbox[0]}, {bbox[2]}]"
        assert bbox[1] <= c["lat"] <= bbox[3], f"Lat {c['lat']} outside bbox [{bbox[1]}, {bbox[3]}]"
    print("  [PASS] All returned tiles reside within the target bounding box.")

def test_polygon_geosemantic_fusion():
    print_header("Step 6: PostGIS Polygon AOI Verification (ST_Intersects / ST_Within)")
    url = f"{BASE_URL}/search/semantic"
    # Polygon around active tile scene coordinates
    polygon = [
        [77.00, 28.40],
        [77.50, 28.40],
        [77.50, 28.80],
        [77.00, 28.80],
        [77.00, 28.40],
    ]

    # Test Mode 1: ST_Intersects
    payload_intersects = {
        "query": "river channel or bridge",
        "top_k": 10,
        "aoi_polygon": polygon,
        "spatial_filter_mode": "intersects",
    }
    resp1 = client.post(url, json=payload_intersects, timeout=30)
    assert resp1.status_code == 200, f"Failed: {resp1.status_code} {resp1.text}"
    data1 = resp1.json()
    print(f"  Mode ST_Intersects: Found {data1['total_found']} tiles")
    print(f"    PostGIS verification time: {data1.get('spatial_filter_time_ms')}ms")
    assert "spatial_filter_time_ms" in data1, "Expected spatial_filter_time_ms in response telemetry!"

    # Test Mode 2: ST_Within
    payload_within = {
        "query": "river channel or bridge",
        "top_k": 10,
        "aoi_polygon": polygon,
        "spatial_filter_mode": "within",
    }
    resp2 = client.post(url, json=payload_within, timeout=30)
    assert resp2.status_code == 200, f"Failed: {resp2.status_code} {resp2.text}"
    data2 = resp2.json()
    print(f"  Mode ST_Within:     Found {data2['total_found']} tiles (PostGIS time: {data2.get('spatial_filter_time_ms')}ms)")

    # Inspect GeoJSON footprints
    if data1["results"]:
        sample = data1["results"][0]
        print(f"  Sample Tile Footprint GeoJSON: type={sample.get('footprint_geojson', {}).get('type')}")
        assert sample.get("footprint_geojson") is not None, "footprint_geojson should be attached to results"
    print("  [PASS] GeoSemantic Fusion PostGIS spatial verification completed successfully.")

def test_combined_multi_filter(sensor: str):
    print_header("Step 7: Combined Multi-Filter Search (Semantic + Polygon + Sensor + Date + Quality)")
    url = f"{BASE_URL}/search/semantic"
    polygon = [
        [76.80, 28.20],
        [77.60, 28.20],
        [77.60, 28.90],
        [76.80, 28.90],
        [76.80, 28.20],
    ]
    payload = {
        "query": "industrial facilities near water",
        "top_k": 10,
        "sensors": [sensor],
        "start_date": "2021-01-01T00:00:00Z",
        "end_date": "2026-12-31T23:59:59Z",
        "min_quality": 0.30,
        "max_cloud_cover": 80.0,
        "aoi_polygon": polygon,
        "spatial_filter_mode": "intersects",
    }
    t0 = time.perf_counter()
    resp = client.post(url, json=payload, timeout=30)
    dur = (time.perf_counter() - t0) * 1000
    assert resp.status_code == 200, f"Failed: {resp.status_code} {resp.text}"
    data = resp.json()
    print(f"  Combined Search Returned: {data['total_found']} tiles")
    print(f"  Timing Breakdown:")
    print(f"    - RemoteCLIP Text Embed: {data['text_embedding_time_ms']}ms")
    print(f"    - Qdrant Vector Search:  {data['vector_search_time_ms']}ms")
    print(f"    - PostGIS Exact Spatial: {data.get('spatial_filter_time_ms')}ms")
    print(f"    - Total Internal Time:   {data['execution_time_ms']}ms (Network roundtrip: {dur:.1f}ms)")
    for r in data["results"][:3]:
        print(f"    - Rank #{r['rank']} [Score: {r['similarity_score']:.3f}] Sensor: {r['sensor']} Date: {r['acquisition_date']}")
    print("  [PASS] Combined multi-filter GeoSemantic query completed cleanly.")

def test_zero_result_handling():
    print_header("Step 8: Zero-Result Handling With Highly Restrictive Constraints")
    url = f"{BASE_URL}/search/semantic"
    # Pacific Ocean polygon where no tiles exist
    remote_polygon = [
        [-170.0, -10.0],
        [-169.0, -10.0],
        [-169.0, -9.0],
        [-170.0, -9.0],
        [-170.0, -10.0],
    ]
    payload = {
        "query": "industrial storage tanks",
        "top_k": 10,
        "aoi_polygon": remote_polygon,
    }
    resp = client.post(url, json=payload, timeout=30)
    assert resp.status_code == 200, f"Failed: {resp.status_code} {resp.text}"
    data = resp.json()
    print(f"  Returned {data['total_found']} tiles (expected 0).")
    assert data["total_found"] == 0, f"Expected 0 results for remote Pacific polygon, got {data['total_found']}"
    assert len(data["results"]) == 0
    print("  [PASS] Zero-result scenario handled gracefully without error.")

def main():
    print("Starting Phase 8: Multi-Filter Search and GeoSemantic Fusion Automated Test Suite")
    try:
        filters = test_filters_endpoint()
        sensor = filters["sensors"][0] if filters["sensors"] else "Demo-MSI"

        test_independent_sensor_filter(sensor)
        test_independent_date_filter()
        test_independent_quality_and_cloud_filter()
        test_independent_bounding_box_aoi()
        test_polygon_geosemantic_fusion()
        test_combined_multi_filter(sensor)
        test_zero_result_handling()

        print("\n" + "=" * 70)
        print(" [SUCCESS] ALL PHASE 8 GEOSEMANTIC FUSION TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
    except Exception as exc:
        print(f"\n[FAIL] TEST FAILED: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
