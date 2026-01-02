# Pre-Commit Workflow

Run this workflow before every commit to ensure code quality and catch regressions.

## Steps

### 1. Start the Inference Server
```bash
./server/start.sh
```
Wait for server to be ready (check logs).

### 2. Run Tests First (Warms up models)
```bash
cd /Users/idofrizler/Git/laundromat && source venv/bin/activate
python -m pytest testing/test_pair_matching.py::TestPairMatching -v --tb=short
```

**Expected Results:**
- At least 25 tests passing
- No more than 5 tests failing
- Check for regressions against previous run

### 3. Run Profiling (After model warmup)
```bash
python scripts/profile_inference.py -s https://localhost:8443 -v ./laundry_pile.mp4 --frames 10 --insecure
```

**Timing Baselines to Watch:**
- SAM3 set_image: ~200-350ms
- SAM3 sock inference: ~100-200ms  
- False-positive filtering: ~100-150ms
- ResNet feature extraction: ~50-150ms
- Total server time: ~600-850ms

If timing significantly exceeds these ranges (>50% regression), investigate before committing.

### 4. Shutdown Server (Optional)
```bash
pkill -f "uvicorn app:app"
```

## Regression Checklist

Before committing, verify:
- [ ] Tests: No new failures compared to previous run
- [ ] Tests: Pass count >= 25
- [ ] Profiling: Total inference time < 850ms
- [ ] Profiling: No individual step >1.5x baseline
