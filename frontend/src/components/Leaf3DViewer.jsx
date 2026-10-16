import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import './Leaf3DViewer.css';

/**
 * Checks whether WebGL rendering context is supported in the current environment.
 */
function isWebGLAvailable() {
  try {
    const canvas = document.createElement('canvas');
    return !!(
      window.WebGLRenderingContext &&
      (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    );
  } catch {
    return false;
  }
}

/**
 * Procedurally generates a realistic organic leaf texture with midrib and fine venation.
 */
function createLeafTexture(isHealthy, cropName) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d');

  // Base background gradient: rich chlorophyll green or subtle chlorotic tint
  const baseGrad = ctx.createLinearGradient(0, 0, 512, 1024);
  if (isHealthy) {
    baseGrad.addColorStop(0, '#2d7a36');   // deep forest green near tip
    baseGrad.addColorStop(0.5, '#3b9247'); // vibrant green mid-blade
    baseGrad.addColorStop(1, '#27682f');   // darker near stem
  } else {
    // Subtle conceptual rustic mottling for diseased condition (distinct from Grad-CAM)
    baseGrad.addColorStop(0, '#667722');
    baseGrad.addColorStop(0.3, '#73832c');
    baseGrad.addColorStop(0.7, '#4e6b2c');
    baseGrad.addColorStop(1, '#334e1c');
  }
  ctx.fillStyle = baseGrad;
  ctx.fillRect(0, 0, 512, 1024);

  // Subtle cellular micro-texture
  ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
  for (let i = 0; i < 4000; i++) {
    const rx = Math.random() * 512;
    const ry = Math.random() * 1024;
    ctx.fillRect(rx, ry, 2, 2);
  }

  // Draw central main vein (midrib)
  ctx.strokeStyle = isHealthy ? 'rgba(180, 230, 160, 0.65)' : 'rgba(210, 205, 140, 0.65)';
  ctx.lineWidth = 14;
  ctx.beginPath();
  ctx.moveTo(256, 1024);
  ctx.quadraticCurveTo(254, 512, 256, 60);
  ctx.stroke();

  // Secondary lateral branching veins
  ctx.lineWidth = 3.5;
  ctx.strokeStyle = isHealthy ? 'rgba(170, 220, 150, 0.45)' : 'rgba(195, 190, 130, 0.45)';
  const numVeinPairs = 10;
  for (let i = 1; i <= numVeinPairs; i++) {
    const yOrigin = 950 - i * 85;
    const angleOffset = 70 + (i * 2);

    // Left vein
    ctx.beginPath();
    ctx.moveTo(256, yOrigin);
    ctx.quadraticCurveTo(160, yOrigin - 40, 40, yOrigin - angleOffset);
    ctx.stroke();

    // Right vein
    ctx.beginPath();
    ctx.moveTo(256, yOrigin);
    ctx.quadraticCurveTo(352, yOrigin - 40, 472, yOrigin - angleOffset);
    ctx.stroke();
  }

  // Conceptual surface variation if diseased (explicitly stylized)
  if (!isHealthy) {
    ctx.fillStyle = 'rgba(130, 90, 30, 0.18)';
    for (let s = 0; s < 12; s++) {
      const sx = 120 + Math.random() * 272;
      const sy = 250 + Math.random() * 500;
      const sr = 12 + Math.random() * 24;
      ctx.beginPath();
      ctx.arc(sx, sy, sr, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  return texture;
}

/**
 * Creates 3D leaf blade geometry with natural curvature and veins.
 */
function createCurvedLeafGeometry(crop) {
  // Shape outline: tapered tip, gentle curved margins, petiole base
  const shape = new THREE.Shape();
  
  // Adjust aspect ratio slightly for corn vs broad leaves (tomato/potato/apple)
  const isCorn = (crop || '').toLowerCase().includes('corn');
  const widthScale = isCorn ? 0.65 : 1.15;
  const heightScale = isCorn ? 1.4 : 1.0;

  shape.moveTo(0, -1.8 * heightScale);
  // Left margin curve up to tip
  shape.bezierCurveTo(
    -0.85 * widthScale, -1.2 * heightScale,
    -1.25 * widthScale, 0.2 * heightScale,
    -0.8 * widthScale, 1.3 * heightScale
  );
  shape.bezierCurveTo(
    -0.5 * widthScale, 1.7 * heightScale,
    -0.15 * widthScale, 1.95 * heightScale,
    0, 2.1 * heightScale // tip
  );

  // Right margin curve down to petiole
  shape.bezierCurveTo(
    0.15 * widthScale, 1.95 * heightScale,
    0.5 * widthScale, 1.7 * heightScale,
    0.8 * widthScale, 1.3 * heightScale
  );
  shape.bezierCurveTo(
    1.25 * widthScale, 0.2 * heightScale,
    0.85 * widthScale, -1.2 * heightScale,
    0, -1.8 * heightScale
  );

  // Extrude shape with thin depth and bevel for organic thickness
  const extrudeSettings = {
    steps: 1,
    depth: 0.02,
    bevelEnabled: true,
    bevelThickness: 0.015,
    bevelSize: 0.015,
    bevelSegments: 3,
  };

  const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);

  // Curve the geometry: add natural longitudinal arch and lateral cupping
  const position = geometry.attributes.position;
  const v = new THREE.Vector3();

  for (let i = 0; i < position.count; i++) {
    v.fromBufferAttribute(position, i);

    // Longitudinal curvature (slight downward dip toward the tip and stem)
    const yNorm = (v.y / (2.1 * heightScale));
    const arch = Math.cos(yNorm * Math.PI * 0.5) * 0.18;

    // Lateral cupping (depressed central midrib gutter with raised/cupped margins)
    const xDist = Math.abs(v.x);
    const cup = Math.pow(xDist / (1.2 * widthScale), 2) * 0.16;

    // Subtle natural edge wave
    const wave = Math.sin(v.y * 5.0) * 0.03 * (xDist > 0.4 ? 1 : 0);

    v.z += arch + cup + wave;
    position.setXYZ(i, v.x, v.y, v.z);
  }

  geometry.computeVertexNormals();
  return geometry;
}

/**
 * Creates raised midrib spine tube geometry.
 */
function createMidribGeometry(crop) {
  const isCorn = (crop || '').toLowerCase().includes('corn');
  const heightScale = isCorn ? 1.4 : 1.0;

  const points = [
    new THREE.Vector3(0, -2.1 * heightScale, 0.03),
    new THREE.Vector3(0, -1.0 * heightScale, 0.10),
    new THREE.Vector3(0, 0.2 * heightScale, 0.14),
    new THREE.Vector3(0, 1.2 * heightScale, 0.11),
    new THREE.Vector3(0, 2.05 * heightScale, 0.04),
  ];

  const curve = new THREE.CatmullRomCurve3(points);
  return new THREE.TubeGeometry(curve, 32, 0.035, 8, false);
}

export default function Leaf3DViewer({
  crop = 'Crop',
  isHealthy = true,
  predictedClass = '',
  displayName = 'Crop Leaf',
}) {
  const mountRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const animationFrameIdRef = useRef(null);
  const leafGroupRef = useRef(null);

  const [hasWebGL, setHasWebGL] = useState(true);
  const [initError, setInitError] = useState(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const [isInteracting, setIsInteracting] = useState(false);

  // Initial camera position for Reset View
  const initialCameraPos = { x: 0, y: 0.2, z: 4.2 };

  // ── Setup Three.js Scene ───────────────────────────────────────────────────
  useEffect(() => {
    if (!isWebGLAvailable()) {
      setHasWebGL(false);
      return;
    }

    const container = mountRef.current;
    if (!container) return;

    let renderer, scene, camera, controls, leafGroup;

    try {
      const width = container.clientWidth || 400;
      const height = container.clientHeight || 340;

      // 1. Scene & Camera
      scene = new THREE.Scene();
      sceneRef.current = scene;

      camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 50);
      camera.position.set(initialCameraPos.x, initialCameraPos.y, initialCameraPos.z);
      cameraRef.current = camera;

      // 2. WebGL Renderer with anti-aliasing and soft background
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance',
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      container.appendChild(renderer.domElement);
      rendererRef.current = renderer;

      // 3. Orbit Controls
      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.enableZoom = true;
      controls.minDistance = 2.0;
      controls.maxDistance = 7.5;
      controls.autoRotate = autoRotate;
      controls.autoRotateSpeed = 1.2;
      controls.target.set(0, 0, 0);

      controls.addEventListener('start', () => setIsInteracting(true));
      controls.addEventListener('end', () => setIsInteracting(false));
      controlsRef.current = controls;

      // 4. Studio Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 1.4);
      scene.add(ambientLight);

      // Key light (soft sun angle highlights leaf blade contours)
      const keyLight = new THREE.DirectionalLight(0xfffbe8, 2.2);
      keyLight.position.set(3, 4, 3.5);
      keyLight.castShadow = true;
      scene.add(keyLight);

      // Fill light (cool soft fill for underside)
      const fillLight = new THREE.DirectionalLight(0xddf0e5, 1.2);
      fillLight.position.set(-3, -2, 2.5);
      scene.add(fillLight);

      // Subtle rim backlight for contour glow
      const rimLight = new THREE.DirectionalLight(0x88cc99, 1.0);
      rimLight.position.set(0, 3, -3);
      scene.add(rimLight);

      // 5. Leaf Group & Procedural Mesh
      leafGroup = new THREE.Group();
      scene.add(leafGroup);
      leafGroupRef.current = leafGroup;

      const leafTexture = createLeafTexture(isHealthy, crop);
      const leafMaterial = new THREE.MeshStandardMaterial({
        map: leafTexture,
        roughness: 0.52,
        metalness: 0.04,
        side: THREE.DoubleSide,
        shadowSide: THREE.DoubleSide,
      });

      const leafGeometry = createCurvedLeafGeometry(crop);
      const leafMesh = new THREE.Mesh(leafGeometry, leafMaterial);
      leafMesh.castShadow = true;
      leafMesh.receiveShadow = true;
      leafGroup.add(leafMesh);

      // Central raised midrib spine
      const midribGeometry = createMidribGeometry(crop);
      const midribMaterial = new THREE.MeshStandardMaterial({
        color: isHealthy ? 0x88cc77 : 0xa6a066,
        roughness: 0.6,
        metalness: 0.02,
      });
      const midribMesh = new THREE.Mesh(midribGeometry, midribMaterial);
      leafGroup.add(midribMesh);

      // Slight initial presentation tilt
      leafGroup.rotation.x = 0.15;
      leafGroup.rotation.y = -0.2;

      // 6. Animation Loop
      const animate = () => {
        animationFrameIdRef.current = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      };
      animate();

      // 7. Responsive Resize Observer
      const handleResize = () => {
        if (!container || !renderer || !camera) return;
        const newWidth = container.clientWidth;
        const newHeight = container.clientHeight;
        if (newWidth > 0 && newHeight > 0) {
          camera.aspect = newWidth / newHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(newWidth, newHeight);
        }
      };

      const resizeObserver = new ResizeObserver(handleResize);
      resizeObserver.observe(container);

      // 8. Cleanup on Unmount
      return () => {
        resizeObserver.disconnect();
        if (animationFrameIdRef.current) {
          cancelAnimationFrame(animationFrameIdRef.current);
        }
        if (controls) controls.dispose();
        if (renderer && renderer.domElement && container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement);
          renderer.dispose();
        }
        leafGeometry.dispose();
        leafMaterial.dispose();
        if (leafTexture) leafTexture.dispose();
        midribGeometry.dispose();
        midribMaterial.dispose();
      };
    } catch (err) {
      console.error('Three.js Leaf3DViewer initialization failed:', err);
      setInitError('Failed to initialize 3D WebGL scene.');
    }
  }, [crop, isHealthy]);

  // Sync autoRotate state with controls
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = autoRotate;
    }
  }, [autoRotate]);

  // ── Reset View Handler ─────────────────────────────────────────────────────
  const handleResetView = useCallback(() => {
    if (cameraRef.current && controlsRef.current && leafGroupRef.current) {
      cameraRef.current.position.set(initialCameraPos.x, initialCameraPos.y, initialCameraPos.z);
      controlsRef.current.target.set(0, 0, 0);
      leafGroupRef.current.rotation.set(0.15, -0.2, 0);
      controlsRef.current.update();
    }
  }, []);

  // ── WebGL Fallback View ────────────────────────────────────────────────────
  if (!hasWebGL || initError) {
    return (
      <div className="leaf-3d-card card fade-in" aria-label="3D leaf viewer unavailable">
        <div className="leaf-3d-card__header">
          <div>
            <span className="leaf-3d-badge">Interactive Reference</span>
            <h3 className="leaf-3d-title">3D Leaf Anatomy Visualization</h3>
          </div>
        </div>
        <div className="leaf-3d-fallback">
          <span className="leaf-3d-fallback__icon" aria-hidden="true">🍃</span>
          <p className="leaf-3d-fallback__text">
            3D WebGL visualization is unavailable on this device or browser.
          </p>
          <span className="leaf-3d-fallback__sub">
            The AI diagnostic results and Grad-CAM explainability above remain fully operational.
          </span>
        </div>
      </div>
    );
  }

  // ── Full Interactive 3D Viewer ─────────────────────────────────────────────
  return (
    <div className="leaf-3d-card card fade-in" aria-label="Interactive 3D Leaf Viewer">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="leaf-3d-card__header">
        <div className="leaf-3d-card__header-info">
          <div className="leaf-3d-badges-row">
            <span className="leaf-3d-badge">Interactive 3D Reference</span>
            <span className={`leaf-3d-status-pill ${isHealthy ? 'pill-healthy' : 'pill-diseased'}`}>
              {isHealthy ? '🌿 Healthy Foliage Model' : '🍂 Pathological Specimen Model'}
            </span>
          </div>
          <h3 className="leaf-3d-title">
            3D Leaf Morphology & Vascular Anatomy
          </h3>
          <p className="leaf-3d-subtitle">
            Rotate and inspect a stylized 3D representation of a <strong>{crop}</strong> leaf blade and its vascular architecture.
          </p>
        </div>

        {/* View Controls Toolbar */}
        <div className="leaf-3d-toolbar" role="toolbar" aria-label="3D Viewer Controls">
          <button
            type="button"
            className={`toolbar-btn ${autoRotate ? 'toolbar-btn--active' : ''}`}
            onClick={() => setAutoRotate((prev) => !prev)}
            title={autoRotate ? 'Pause auto-rotation' : 'Start auto-rotation'}
            aria-pressed={autoRotate}
          >
            {autoRotate ? '⏸ Pause Spin' : '▶ Auto-Spin'}
          </button>
          <button
            type="button"
            className="toolbar-btn btn-reset-view"
            onClick={handleResetView}
            title="Reset leaf orientation and camera zoom"
          >
            ↺ Reset View
          </button>
        </div>
      </div>

      {/* ── Canvas Viewport ─────────────────────────────────────────────────── */}
      <div className="leaf-3d-viewport-wrapper">
        <div
          ref={mountRef}
          className="leaf-3d-viewport"
          aria-label="Interactive 3D canvas. Click and drag to rotate, scroll to zoom."
        />

        {/* Interactive Overlay Hints */}
        <div className="leaf-3d-hud" aria-hidden="true">
          <span className="hud-chip">🖱️ Drag to Rotate</span>
          <span className="hud-chip">🔍 Scroll to Zoom</span>
        </div>
      </div>

      {/* ── Explicit Technical Honesty Disclaimer ──────────────────────────── */}
      <div className="leaf-3d-disclaimer" role="note">
        <span className="disclaimer-icon" aria-hidden="true">ℹ️</span>
        <div className="disclaimer-content">
          <strong>Conceptual Reference Model:</strong> This interactive 3D leaf illustrates generalized crop foliage morphology, venation, and natural curvature. It is an educational reference and <em>does not reconstruct the uploaded photograph or localize disease lesions</em>. For scientific AI explainability showing where the neural network focused, consult the <strong>Grad-CAM Visual Attention Heatmap</strong> displayed above.
        </div>
      </div>
    </div>
  );
}
