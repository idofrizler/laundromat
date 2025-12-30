/**
 * Laundromat Web Client
 * High-performance implementation mimicking Python client logic.
 * Features:
 * - Synchronized Inference & Tracking
 * - Optical Flow Lag Compensation
 * - Resolution Kill Switch (Downscaled Processing)
 * - Zero-Allocation Render Loop
 */

class Tracker {
    constructor() {
        this.cvReady = false;
        
        // Configuration: Process at low res for speed
        this.trackWidth = 320;
        this.trackHeight = 240;
        
        // State
        this.scaleX = 1;
        this.scaleY = 1;
        this.frameCount = 0;
        
        // Memory: Pre-allocated OpenCV Mats
        this.mats = {
            src: null,      // RGBA input
            prev: null,     // Previous Gray
            curr: null,     // Current Gray
            snap: null,     // Snapshot Gray (for lag comp)
            p0: null,       // Points to track
            p1: null,       // Tracked points
            st: null,       // Status
            err: null,      // Error
            mask: null      // Temp mask
        };
        
        // Helper canvas for downscaling
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.trackWidth;
        this.canvas.height = this.trackHeight;
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    }
    
    init() {
        if (typeof cv !== 'undefined' && cv.Mat && !this.cvReady) {
            try {
                this.mats.src = new cv.Mat(this.trackHeight, this.trackWidth, cv.CV_8UC4);
                this.mats.prev = new cv.Mat(this.trackHeight, this.trackWidth, cv.CV_8UC1);
                this.mats.curr = new cv.Mat(this.trackHeight, this.trackWidth, cv.CV_8UC1);
                this.mats.snap = new cv.Mat(this.trackHeight, this.trackWidth, cv.CV_8UC1);
                this.mats.p0 = new cv.Mat();
                this.mats.p1 = new cv.Mat();
                this.mats.st = new cv.Mat();
                this.mats.err = new cv.Mat();
                this.mats.mask = new cv.Mat();
                this.cvReady = true;
                console.log('[TRACKER] OpenCV Initialized');
            } catch (e) {
                console.error('[TRACKER] Init failed:', e);
            }
        }
    }
    
    // Process a frame (Video/Canvas -> Gray Mat)
    process(source) {
        if (!this.cvReady) return;
        
        // Update scale factors
        const srcW = source.videoWidth || source.width;
        const srcH = source.videoHeight || source.height;
        this.scaleX = srcW / this.trackWidth;
        this.scaleY = srcH / this.trackHeight;
        
        // Swap buffers
        if (this.frameCount > 0) {
            this.mats.curr.copyTo(this.mats.prev);
        }
        
        // Downscale draw
        this.ctx.drawImage(source, 0, 0, this.trackWidth, this.trackHeight);
        const imgData = this.ctx.getImageData(0, 0, this.trackWidth, this.trackHeight);
        this.mats.src.data.set(imgData.data);
        cv.cvtColor(this.mats.src, this.mats.curr, cv.COLOR_RGBA2GRAY);
        this.frameCount++;
    }
    
    // Save current state as snapshot (for lag compensation)
    snapshot() {
        if (this.cvReady) this.mats.curr.copyTo(this.mats.snap);
    }
    
    // Calculate motion: Snapshot -> Current Frame
    getLagCompensation() {
        if (!this.cvReady) return { x: 0, y: 0 };
        
        try {
            // 1. Find features in the OLD snapshot
            cv.goodFeaturesToTrack(
                this.mats.snap, this.mats.p0, 
                50, 0.3, 7, this.mats.mask, 7
            );
            
            if (this.mats.p0.rows === 0) return { x: 0, y: 0 };
            
            // 2. Track to CURRENT frame
            const winSize = new cv.Size(21, 21);
            const criteria = new cv.TermCriteria(cv.TermCriteria_EPS | cv.TermCriteria_COUNT, 30, 0.01);
            
            cv.calcOpticalFlowPyrLK(
                this.mats.snap, this.mats.curr, 
                this.mats.p0, this.mats.p1, 
                this.mats.st, this.mats.err, 
                winSize, 3, criteria
            );
            
            return this._calcAverageShift(this.mats.p0, this.mats.p1, this.mats.st);
        } catch (e) {
            console.error('Lag comp error:', e);
            return { x: 0, y: 0 };
        }
    }
    
    // Track points: Prev Frame -> Current Frame
    trackPoints(points) {
        if (!this.cvReady || !points.length) return { points, shift: {x:0, y:0} };
        
        try {
            // 1. Prepare points (downscale)
            const numPoints = points.length;
            const inputArr = new Float32Array(numPoints * 2);
            for (let i = 0; i < numPoints; i++) {
                inputArr[i*2] = points[i][0] / this.scaleX;
                inputArr[i*2+1] = points[i][1] / this.scaleY;
            }
            
            if (this.mats.p0) this.mats.p0.delete();
            this.mats.p0 = cv.matFromArray(numPoints, 1, cv.CV_32FC2, inputArr);
            
            // 2. Run Flow
            const winSize = new cv.Size(21, 21);
            const criteria = new cv.TermCriteria(cv.TermCriteria_EPS | cv.TermCriteria_COUNT, 30, 0.01);
            
            cv.calcOpticalFlowPyrLK(
                this.mats.prev, this.mats.curr, 
                this.mats.p0, this.mats.p1, 
                this.mats.st, this.mats.err, 
                winSize, 3, criteria
            );
            
            // 3. Extract results
            const newPoints = [];
            const stData = this.mats.st.data;
            const p1Data = this.mats.p1.data32F;
            
            for (let i = 0; i < numPoints; i++) {
                if (stData[i] === 1) {
                    newPoints.push([
                        p1Data[i*2] * this.scaleX,
                        p1Data[i*2+1] * this.scaleY
                    ]);
                }
            }
            
            const shift = this._calcAverageShift(this.mats.p0, this.mats.p1, this.mats.st);
            return { points: newPoints, shift };
        } catch (e) {
            console.error('Track error:', e);
            return { points, shift: {x:0, y:0} };
        }
    }
    
    _calcAverageShift(p0, p1, st) {
        let dx = 0, dy = 0, count = 0;
        const stData = st.data;
        const p0Data = p0.data32F;
        const p1Data = p1.data32F;
        
        for (let i = 0; i < st.rows; i++) {
            if (stData[i] === 1) {
                dx += p1Data[i*2] - p0Data[i*2];
                dy += p1Data[i*2+1] - p0Data[i*2+1];
                count++;
            }
        }
        
        if (count > 0) {
            return { x: (dx / count) * this.scaleX, y: (dy / count) * this.scaleY };
        }
        return { x: 0, y: 0 };
    }
}

class LaundromatClient {
    constructor() {
        // UI Elements
        this.video = document.getElementById('videoElement');
        this.canvas = document.getElementById('overlayCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Core Components
        this.tracker = new Tracker();
        this.captureCanvas = document.createElement('canvas');
        this.captureCtx = this.captureCanvas.getContext('2d', { willReadFrequently: true });
        
        // Recording: Composite canvas (video + overlay)
        this.recordCanvas = document.createElement('canvas');
        this.recordCtx = this.recordCanvas.getContext('2d');
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.isRecording = false;
        
        // State
        this.isProcessing = false;
        this.pairs = [];
        this.basketMasks = [];  // Detected basket masks with pre-rendered canvases
        this.fps = 0;
        this.lastFrameTime = 0;
        this.inferMode = 'auto';  // 'auto' or 'manual'
        
        // Compute server URL based on current location
        this.serverUrl = `${location.protocol}//${location.hostname}:${location.protocol === 'https:' ? '8443' : '8080'}`;
        
        this.bindEvents();
    }
    
    bindEvents() {
        document.getElementById('startBtn').onclick = () => this.start();
        document.getElementById('stopBtn').onclick = () => this.stop();
        document.getElementById('recordBtn').onclick = () => this.toggleRecording();
        document.getElementById('refreshInterval').onchange = () => this.updateInferenceInterval();
        document.getElementById('inferMode').onchange = (e) => this.setInferMode(e.target.value);
        this.video.onloadedmetadata = () => this.onVideoReady();
        
        // Tap to detect in manual mode (on video/canvas area only, not controls)
        document.getElementById('container').onclick = (e) => {
            // Don't trigger if clicking on controls or other UI
            if (e.target.closest('#ui-layer')) return;
            if (this.inferMode === 'manual' && this.video.srcObject) {
                this.runInference();
            }
        };
    }
    
    setInferMode(mode) {
        this.inferMode = mode;
        console.log(`[MODE] Switched to ${mode} mode`);
        
        // Show/hide interval input based on mode
        const intervalGroup = document.getElementById('autoIntervalGroup');
        intervalGroup.style.display = mode === 'auto' ? 'flex' : 'none';
        
        // Clear or restart interval based on mode
        if (this.inferInterval) {
            clearInterval(this.inferInterval);
            this.inferInterval = null;
        }
        
        if (mode === 'auto' && this.video.srcObject) {
            const interval = parseInt(document.getElementById('refreshInterval').value) || 2;
            this.inferInterval = setInterval(() => this.runInference(), interval * 1000);
        }
    }
    
    updateInferenceInterval() {
        // Clear existing interval
        if (this.inferInterval) {
            clearInterval(this.inferInterval);
        }
        
        // Only restart if video is active
        if (this.video.srcObject) {
            const interval = parseInt(document.getElementById('refreshInterval').value) || 2;
            console.log(`[INFER] Interval updated to ${interval}s`);
            this.inferInterval = setInterval(() => this.runInference(), interval * 1000);
        }
    }
    
    toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            this.startRecording();
        }
    }
    
    startRecording() {
        // Setup record canvas size
        this.recordCanvas.width = this.video.videoWidth;
        this.recordCanvas.height = this.video.videoHeight;
        
        // Get stream from canvas
        const stream = this.recordCanvas.captureStream(30); // 30 FPS
        
        // Determine supported mime type
        const mimeTypes = [
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm',
            'video/mp4'
        ];
        let mimeType = '';
        for (const type of mimeTypes) {
            if (MediaRecorder.isTypeSupported(type)) {
                mimeType = type;
                break;
            }
        }
        
        if (!mimeType) {
            alert('No supported video format found for recording');
            return;
        }
        
        this.recordedChunks = [];
        this.mediaRecorder = new MediaRecorder(stream, { 
            mimeType,
            videoBitsPerSecond: 8000000 // 8 Mbps for high quality
        });
        
        this.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                this.recordedChunks.push(e.data);
            }
        };
        
        this.mediaRecorder.onstop = () => {
            const blob = new Blob(this.recordedChunks, { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `laundromat-${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.webm`;
            a.click();
            URL.revokeObjectURL(url);
        };
        
        this.mediaRecorder.start(100); // Collect data every 100ms
        this.isRecording = true;
        
        const btn = document.getElementById('recordBtn');
        btn.textContent = '⏹ Stop';
        btn.classList.add('recording');
        console.log('[RECORD] Started');
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        this.isRecording = false;
        
        const btn = document.getElementById('recordBtn');
        btn.textContent = '⏺ Record';
        btn.classList.remove('recording');
        console.log('[RECORD] Stopped');
    }
    
    async start() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
            });
            this.video.srcObject = stream;
            this.video.play();
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('recordBtn').disabled = false;
        } catch (e) {
            alert('Camera error: ' + e.message);
        }
    }
    
    stop() {
        // Stop recording if active
        if (this.isRecording) {
            this.stopRecording();
        }
        
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(t => t.stop());
            this.video.srcObject = null;
        }
        clearInterval(this.inferInterval);
        cancelAnimationFrame(this.animId);
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('recordBtn').disabled = true;
    }
    
    onVideoReady() {
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        this.captureCanvas.width = this.video.videoWidth;
        this.captureCanvas.height = this.video.videoHeight;
        
        this.tracker.init();
        
        // Start inference loop based on current mode
        this.inferMode = document.getElementById('inferMode').value;
        if (this.inferMode === 'auto') {
            const interval = parseInt(document.getElementById('refreshInterval').value) || 2;
            this.inferInterval = setInterval(() => this.runInference(), interval * 1000);
        }
        
        // Start render loop
        this.render();
    }
    
    showDetectingOverlay(show) {
        const overlay = document.getElementById('detectingOverlay');
        overlay.style.display = show ? 'flex' : 'none';
    }
    
    async runInference() {
        if (this.isProcessing) return;
        this.isProcessing = true;
        
        // Show overlay only in manual mode
        if (this.inferMode === 'manual') {
            this.showDetectingOverlay(true);
        }
        
        try {
            // 1. Capture Frame & Sync Tracker
            this.captureCtx.drawImage(this.video, 0, 0);
            this.tracker.process(this.captureCanvas); // Update tracker with THIS frame
            this.tracker.snapshot(); // Save state for lag comp
            
            // 2. Send to Server
            const blob = await new Promise(r => this.captureCanvas.toBlob(r, 'image/jpeg', 0.8));
            
            const formData = new FormData();
            formData.append('frame', blob, 'frame.jpg');
            const topN = document.getElementById('topNPairs').value;
            const excludeBasket = document.getElementById('excludeBasket').checked;

            const response = await fetch(
                `${this.serverUrl}/infer?top_n_pairs=${topN}&exclude_basket=${excludeBasket}`,
                { method: 'POST', body: formData }
            );
            
            if (!response.ok) throw new Error(response.statusText);
            const result = await response.json();
            
            // 3. Lag Compensation (Snapshot -> Current)
            // Note: Tracker has been updating in render loop while we waited
            const lagShift = this.tracker.getLagCompensation();
            console.log(`[INFER] Lag Comp: ${lagShift.x.toFixed(1)}, ${lagShift.y.toFixed(1)}`);
            
            // 4. Process Results
            this.pairs = result.pairs_data.map(item => {
                const mask = this.decodeMaskRLE(item.mask_rle);
                
                // Calculate scale factors (mask coords -> video coords)
                const scaleX = this.video.videoWidth / mask.width;
                const scaleY = this.video.videoHeight / mask.height;
                
                // Normalize and scale points from mask coords to video coords
                let points = (item.points || []).map(p => {
                    const pt = Array.isArray(p[0]) ? p[0] : p;
                    return [pt[0] * scaleX, pt[1] * scaleY];
                });
                
                // Apply Lag Shift (in video coordinates)
                points = points.map(p => [p[0] + lagShift.x, p[1] + lagShift.y]);
                
                const p = {
                    mask: mask,
                    box: item.box,
                    label: item.label,
                    color: item.color,
                    points: points,
                    offset: { x: lagShift.x, y: lagShift.y }
                };
                this.preRenderMask(p);
                return p;
            });
            
            // Process basket masks (RLE-encoded) with tracking
            this.basketMasks = (result.basket_masks || []).map(maskRle => {
                const mask = this.decodeMaskRLE(maskRle);
                
                // Calculate scale factors (mask coords -> video coords)
                const scaleX = this.video.videoWidth / mask.width;
                const scaleY = this.video.videoHeight / mask.height;
                
                // Generate tracking points from mask (sample points inside the mask)
                const points = this.generateTrackingPointsFromMask(mask, scaleX, scaleY, lagShift);
                
                return {
                    mask: mask,
                    canvas: this.preRenderBasketMask(mask),
                    points: points,
                    offset: { x: lagShift.x, y: lagShift.y }
                };
            });
            console.log(`[INFER] ${this.basketMasks.length} basket(s) detected with tracking`);
            
            // Update UI
            document.getElementById('totalSocks').textContent = result.total_socks_detected;
            document.getElementById('pairsMatched').textContent = Math.floor(result.pairs_data.length / 2);
            document.getElementById('basketCount').textContent = this.basketMasks.length;
            document.getElementById('inferenceTime').textContent = Math.round(result.inference_time_ms) + 'ms';
            document.getElementById('statusDot').className = 'status-dot connected';
            document.getElementById('statusText').textContent = 'Connected';

        } catch (e) {
            console.error(e);
            document.getElementById('statusDot').className = 'status-dot';
            document.getElementById('statusText').textContent = 'Error';
        } finally {
            this.isProcessing = false;
            // Hide overlay (only visible in manual mode anyway)
            this.showDetectingOverlay(false);
        }
    }
    
    decodeMaskRLE(rle) {
        const counts = rle.counts;
        const [h, w] = rle.size;
        const flat = new Uint8Array(h * w);
        let pos = 0;
        let val = 0;
        for (const count of counts) {
            flat.fill(val * 255, pos, pos + count);
            pos += count;
            val = 1 - val;
        }
        // Column-major to Row-major
        const mask = new Uint8Array(h * w);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                mask[y * w + x] = flat[x * h + y];
            }
        }
        return { data: mask, width: w, height: h };
    }
    
    preRenderMask(pair) {
        const { mask, color } = pair;
        const c = document.createElement('canvas');
        c.width = mask.width;
        c.height = mask.height;
        const ctx = c.getContext('2d');
        const img = ctx.createImageData(mask.width, mask.height);
        
        for (let i = 0; i < mask.data.length; i++) {
            if (mask.data[i] > 0) {
                const idx = i * 4;
                img.data[idx] = color[0];
                img.data[idx+1] = color[1];
                img.data[idx+2] = color[2];
                img.data[idx+3] = 160;
            }
        }
        ctx.putImageData(img, 0, 0);
        pair.maskCanvas = c;
    }
    
    preRenderBasketMask(mask) {
        const c = document.createElement('canvas');
        c.width = mask.width;
        c.height = mask.height;
        const ctx = c.getContext('2d');
        const img = ctx.createImageData(mask.width, mask.height);
        
        // Soft grey with moderate transparency
        const r = 128, g = 128, b = 128, a = 100;
        
        for (let i = 0; i < mask.data.length; i++) {
            if (mask.data[i] > 0) {
                const idx = i * 4;
                img.data[idx] = r;
                img.data[idx+1] = g;
                img.data[idx+2] = b;
                img.data[idx+3] = a;
            }
        }
        ctx.putImageData(img, 0, 0);
        return c;
    }
    
    generateTrackingPointsFromMask(mask, scaleX, scaleY, lagShift) {
        // Sample points uniformly from inside the mask for tracking
        const points = [];
        const maxPoints = 30;
        const step = Math.max(5, Math.floor(Math.min(mask.width, mask.height) / 10));
        
        for (let y = step; y < mask.height - step && points.length < maxPoints; y += step) {
            for (let x = step; x < mask.width - step && points.length < maxPoints; x += step) {
                if (mask.data[y * mask.width + x] > 0) {
                    // Scale to video coordinates and apply lag shift
                    points.push([
                        x * scaleX + lagShift.x,
                        y * scaleY + lagShift.y
                    ]);
                }
            }
        }
        
        return points;
    }
    
    render() {
        this.animId = requestAnimationFrame(() => this.render());
        
        // FPS Calc
        const now = performance.now();
        const delta = now - this.lastFrameTime;
        if (delta >= 1000) {
            this.fps = (this.frameCount || 0) * 1000 / delta;
            this.frameCount = 0;
            this.lastFrameTime = now;
        }
        this.frameCount = (this.frameCount || 0) + 1;
        
        // 1. Update Tracker
        this.tracker.process(this.video);
        
        // 2. Track Sock Pairs
        for (const p of this.pairs) {
            if (p.points && p.points.length) {
                const res = this.tracker.trackPoints(p.points);
                p.points = res.points;
                p.offset.x += res.shift.x;
                p.offset.y += res.shift.y;
            }
        }
        
        // 3. Track Baskets
        for (const basket of this.basketMasks) {
            if (basket.points && basket.points.length) {
                const res = this.tracker.trackPoints(basket.points);
                basket.points = res.points;
                basket.offset.x += res.shift.x;
                basket.offset.y += res.shift.y;
            }
        }
        
        // 3. Draw
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        for (const p of this.pairs) {
            if (!p.maskCanvas) continue;
            
            // Calculate scale factors for this mask (mask coords -> canvas coords)
            const scaleX = this.canvas.width / p.mask.width;
            const scaleY = this.canvas.height / p.mask.height;
            
            // offset is in video/canvas coordinates, so use it directly for translation
            this.ctx.save();
            this.ctx.translate(p.offset.x, p.offset.y);
            this.ctx.drawImage(p.maskCanvas, 0, 0, this.canvas.width, this.canvas.height);
            this.ctx.restore();
            
            // Label - box coordinates are in mask space, need to scale to canvas
            // Then add offset which is already in canvas coordinates
            const cx = (p.box[0] + p.box[2]) / 2 * scaleX + p.offset.x;
            const cy = (p.box[1] + p.box[3]) / 2 * scaleY + p.offset.y;
            
            this.ctx.font = 'bold 24px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillStyle = `rgb(${p.color.join(',')})`;
            this.ctx.strokeStyle = 'black';
            this.ctx.lineWidth = 3;
            this.ctx.strokeText(p.label, cx, cy);
            this.ctx.fillText(p.label, cx, cy);
        }
        
        // Draw basket masks with soft grey overlay and label (with tracking offset)
        for (const basket of this.basketMasks) {
            if (!basket.canvas) continue;
            
            // Calculate scale factors for this mask (mask coords -> canvas coords)
            const scaleX = this.canvas.width / basket.mask.width;
            const scaleY = this.canvas.height / basket.mask.height;
            
            // Draw the pre-rendered mask scaled to canvas size with offset
            this.ctx.save();
            this.ctx.translate(basket.offset.x, basket.offset.y);
            this.ctx.drawImage(basket.canvas, 0, 0, this.canvas.width, this.canvas.height);
            this.ctx.restore();
            
            // Find center of mass of the mask for label placement
            let sumX = 0, sumY = 0, count = 0;
            for (let y = 0; y < basket.mask.height; y++) {
                for (let x = 0; x < basket.mask.width; x++) {
                    if (basket.mask.data[y * basket.mask.width + x] > 0) {
                        sumX += x;
                        sumY += y;
                        count++;
                    }
                }
            }
            
            if (count > 0) {
                // Scale center to canvas coords and add tracking offset
                const cx = (sumX / count) * scaleX + basket.offset.x;
                const cy = (sumY / count) * scaleY + basket.offset.y;
                
                // Draw "Basket" label with better visibility
                this.ctx.font = 'bold 32px sans-serif';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                
                // Draw text shadow/outline for visibility
                this.ctx.strokeStyle = 'white';
                this.ctx.lineWidth = 5;
                this.ctx.strokeText('🧺 Basket', cx, cy);
                
                this.ctx.fillStyle = '#404040';
                this.ctx.fillText('🧺 Basket', cx, cy);
            }
        }
        
        // FPS
        this.ctx.font = 'bold 20px monospace';
        this.ctx.fillStyle = this.fps > 20 ? '#0f0' : '#f00';
        this.ctx.textAlign = 'right';
        this.ctx.textBaseline = 'alphabetic';
        this.ctx.fillText(this.fps.toFixed(1) + ' FPS', this.canvas.width - 10, 30);
        
        // Recording: Composite video + overlay onto record canvas
        if (this.isRecording) {
            this.recordCtx.drawImage(this.video, 0, 0);
            this.recordCtx.drawImage(this.canvas, 0, 0);
        }
    }
}

// Initialize on page load
window.onload = () => window.app = new LaundromatClient();
