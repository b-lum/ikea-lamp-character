// Body viewer: builds the lamp scene generically from the URDF description the
// server sends on connect, then displays streamed joint/light state. Pure
// display — no state lives here.
import * as THREE from 'three';
import { STLLoader } from './vendor/STLLoader.js';
import { OrbitControls } from './vendor/OrbitControls.js';

// ---------- renderer / scene ----------
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d12);
scene.fog = new THREE.Fog(0x0b0d12, 2.5, 6);

const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 20);
camera.position.set(0.85, 0.65, 1.05);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.35, 0);
controls.enableDamping = true;
controls.maxDistance = 4;

function resize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener('resize', resize);
resize();

// stage
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(3, 64),
  new THREE.MeshStandardMaterial({ color: 0x14171f, roughness: 0.9 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

scene.add(new THREE.HemisphereLight(0x8899bb, 0x14161c, 0.75));
const key = new THREE.DirectionalLight(0xbfd0ff, 0.9);
key.position.set(1.5, 2.5, 1.2);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
scene.add(key);
const rim = new THREE.DirectionalLight(0xff9a5c, 0.4); // warm rim so the black body reads
rim.position.set(-1.8, 1.2, -1.5);
scene.add(rim);

// URDF is Z-up; three.js is Y-up. One root rotation converts everything.
const robotRoot = new THREE.Group();
robotRoot.rotation.x = -Math.PI / 2;
scene.add(robotRoot);
// lampRoot carries the display-only hop offset (root.z in the protocol)
const lampRoot = new THREE.Group();
robotRoot.add(lampRoot);

// ---------- robot construction from description ----------
const stlLoader = new STLLoader();
const jointNodes = {};   // joint name -> { rotor: Group, axis: Vector3 }
let lampLight = null;    // { spot, cone, discs: [materials] }

function eulerFromRPY(rpy) {
  // URDF rpy is extrinsic X-Y-Z == intrinsic Z-Y-X.
  return new THREE.Euler(rpy[0], rpy[1], rpy[2], 'ZYX');
}

// Visual theme ("industrial black / exposed Edison bulb"). Purely a viewer
// concern keyed off URDF material names — the robot model itself is untouched.
const POLE_SCALE = 0.7;                    // slim the arm poles
const POLE_R = 0.025 * POLE_SCALE;         // resulting pole radius
const JOINT_R = POLE_R * 1.15;             // joints 15% thicker than poles
const THEME = {
  fixture_white:  { color: 0x17181c, roughness: 0.6,  metalness: 0.25 }, // matte black body
  fixture_chrome: { color: 0x24262c, roughness: 0.3,  metalness: 0.8  }, // gunmetal joints
  fixture_black:  { color: 0x0a0b0d, roughness: 0.5,  metalness: 0.2  },
  fixture_light:  { color: 0xffb45e, emissive: true },                   // amber glow surfaces
};

function makeMaterial(v) {
  const t = THEME[v.material] ?? { color: new THREE.Color(...v.color.slice(0, 3)).getHex(), roughness: 0.55, metalness: 0.3 };
  if (t.emissive) {
    const c = new THREE.Color(t.color);
    return new THREE.MeshStandardMaterial({ color: c, roughness: 0.4, emissive: c, emissiveIntensity: 1.2 });
  }
  return new THREE.MeshStandardMaterial({ color: t.color, roughness: t.roughness, metalness: t.metalness });
}

function buildBulb() {
  // Exposed Edison bulb: amber glass globe, coiled filament, support wires.
  const R = 0.062; // glass radius (styling choice, independent of URDF marker size)
  const g = new THREE.Group();
  const glass = new THREE.Mesh(
    new THREE.SphereGeometry(R, 40, 28),
    new THREE.MeshPhysicalMaterial({
      color: 0xd8893c, transparent: true, opacity: 0.35, roughness: 0.06,
      metalness: 0, emissive: 0xb05e18, emissiveIntensity: 0.25,
    })
  );

  // coiled filament: helix along the bulb's +X axis
  const coil = [];
  const turns = 7, coilR = 0.016, span = R * 1.0;
  for (let i = 0; i <= 140; i++) {
    const t = i / 140;
    coil.push(new THREE.Vector3(
      -span / 2 + span * t,
      coilR * Math.cos(2 * Math.PI * turns * t),
      coilR * Math.sin(2 * Math.PI * turns * t)
    ));
  }
  const filament = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3(coil), 200, 0.0022, 6),
    new THREE.MeshBasicMaterial({ color: 0xffc070, transparent: true })
  );

  // two support wires running in from the socket (-X) side
  const wireMat = new THREE.MeshStandardMaterial({ color: 0x2b2b2b, roughness: 0.4, metalness: 0.6 });
  const wireGeo = new THREE.CylinderGeometry(0.0012, 0.0012, R * 0.75, 6);
  wireGeo.rotateZ(Math.PI / 2);
  for (const side of [-1, 1]) {
    const wire = new THREE.Mesh(wireGeo, wireMat);
    wire.position.set(-R * 0.55, side * coilR, 0);
    g.add(wire);
  }

  const glow = new THREE.PointLight(0xffb45e, 0.5, 1.4, 2);
  g.add(glass, filament, glow);
  g.userData.bulb = { glassMat: glass.material, filamentMat: filament.material, glow };
  return g;
}

let _woodTex;
function woodTexture() {
  if (_woodTex) return _woodTex;
  const c = document.createElement('canvas');
  c.width = c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#4a3120'; // dark hazel base tone
  g.fillRect(0, 0, 512, 512);
  const shades = ['#3a2517', '#5c3e26', '#6b4a2e', '#43301e', '#7a5636'];
  for (let i = 0; i < 170; i++) {
    g.globalAlpha = 0.1 + Math.random() * 0.25;
    g.fillStyle = shades[i % shades.length];
    g.fillRect(Math.random() * 512, 0, 1 + Math.random() * 4, 512);
  }
  g.globalAlpha = 1;
  _woodTex = new THREE.CanvasTexture(c);
  _woodTex.wrapS = _woodTex.wrapT = THREE.RepeatWrapping;
  _woodTex.colorSpace = THREE.SRGBColorSpace;
  return _woodTex;
}

function buildWoodBase(v) {
  // 3/4-height polished dark-hazel wood base, bottom kept on the ground.
  // The URDF's turntable joint stays at its original height, so a slim stem
  // bridges the gap up to the shoulder hinge (cosmetic only).
  const wrapper = new THREE.Group();
  const h = v.length * 0.75;
  wrapper.position.set(0, 0, h / 2);
  const wood = new THREE.MeshPhysicalMaterial({
    map: woodTexture(), roughness: 0.32, metalness: 0,
    clearcoat: 0.8, clearcoatRoughness: 0.2,
  });
  const base = new THREE.Mesh(new THREE.CylinderGeometry(v.radius, v.radius, h, 64), wood);
  base.rotation.x = Math.PI / 2;
  base.castShadow = true;
  const stem = new THREE.Mesh(
    new THREE.CylinderGeometry(POLE_R * 0.8, POLE_R * 1.1, 0.115 - h, 32),
    new THREE.MeshStandardMaterial({ color: 0x17181c, roughness: 0.6, metalness: 0.25 })
  );
  stem.rotation.x = Math.PI / 2;
  stem.position.set(0, 0, (0.115 + h) / 2 - h / 2); // spans base top -> shoulder hinge
  stem.castShadow = true;
  wrapper.add(base, stem);
  return wrapper;
}

function buildVisual(v, linkName) {
  // Exposed-bulb styling: drop the shade mesh and its inner light disc, and
  // render the emitter marker as a full Edison bulb. Viewer-only decisions —
  // the URDF still describes the same robot.
  if (linkName === 'lamp_head_link' && (v.shape === 'mesh' || v.material === 'fixture_light')) return null;
  if (linkName === 'camera_link') return null; // camera marker nub, exposed once the shade went
  if (linkName === 'base_link') return buildWoodBase(v);

  const wrapper = new THREE.Group();
  wrapper.position.fromArray(v.origin_xyz);
  wrapper.setRotationFromEuler(eulerFromRPY(v.origin_rpy));
  if (linkName === 'light_emitter_link' && v.shape === 'sphere') {
    const bulb = buildBulb();
    wrapper.add(bulb);
    wrapper.userData.bulb = bulb.userData.bulb;
    return wrapper;
  }
  const material = makeMaterial(v);
  let radius = v.radius;
  let length = v.length;
  if (v.material === 'fixture_white' && v.shape === 'cylinder') radius *= POLE_SCALE;
  if (v.material === 'fixture_chrome') {
    radius = v.shape === 'cylinder' ? JOINT_R : v.radius * 0.55;
    // hinge axles lie perpendicular to the arm; trim them to just clear the pole
    const lying = v.origin_rpy[0] !== 0 || v.origin_rpy[1] !== 0;
    if (v.shape === 'cylinder' && lying && length > 0.05) length = 0.048;
  }
  let mesh;
  if (v.shape === 'cylinder') {
    mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length, 40), material);
    mesh.rotation.x = Math.PI / 2; // URDF cylinder axis Z, three's is Y
  } else if (v.shape === 'sphere') {
    mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 32, 24), material);
  } else if (v.shape === 'mesh') {
    mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    stlLoader.load(`/assets/${v.mesh}`, (geo) => { geo.computeVertexNormals(); mesh.geometry = geo; });
  }
  mesh.castShadow = true;
  wrapper.add(mesh);
  if (material.emissive.getHex() !== 0) wrapper.userData.emissiveMaterial = material;
  return wrapper;
}

function buildSocket(headGroup) {
  // black socket bridging the head gimbal to the bare bulb
  const mat = new THREE.MeshStandardMaterial({ color: 0x121317, roughness: 0.45, metalness: 0.5 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.03, 0.085, 32), mat);
  body.rotation.z = Math.PI / 2;
  body.position.set(0.042, 0, 0);
  const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 0.018, 32), mat);
  collar.rotation.z = Math.PI / 2;
  collar.position.set(0.082, 0, 0);
  body.castShadow = collar.castShadow = true;
  headGroup.add(body, collar);
}

function buildRobot(desc) {
  const linkGroups = {};
  for (const [name, visuals] of Object.entries(desc.links)) {
    const g = new THREE.Group();
    g.name = name;
    for (const v of visuals) {
      const w = buildVisual(v, name);
      if (!w) continue;
      g.add(w);
      if (w.userData.emissiveMaterial) {
        (g.userData.emissiveMaterials ??= []).push(w.userData.emissiveMaterial);
      }
      if (w.userData.bulb) g.userData.bulb = w.userData.bulb;
    }
    if (name === 'lamp_head_link') buildSocket(g);
    linkGroups[name] = g;
  }

  const childLinks = new Set(desc.joints.map((j) => j.child));
  for (const j of desc.joints) {
    const origin = new THREE.Group(); // fixed transform from parent link
    origin.position.fromArray(j.origin_xyz);
    origin.setRotationFromEuler(eulerFromRPY(j.origin_rpy));
    const rotor = new THREE.Group(); // rotates about the joint axis
    origin.add(rotor);
    rotor.add(linkGroups[j.child]);
    linkGroups[j.parent].add(origin);
    if (j.type === 'revolute') {
      jointNodes[j.name] = { rotor, axis: new THREE.Vector3().fromArray(j.axis) };
    }
  }
  for (const [name, g] of Object.entries(linkGroups)) {
    if (!childLinks.has(name)) lampRoot.add(g); // root link (base_link)
  }
  buildLampLight(linkGroups);
}

function buildLampLight(linkGroups) {
  const emitter = linkGroups['light_emitter_link'];
  if (!emitter) return;
  const spot = new THREE.SpotLight(0xffe9b0, 0, 3.5, Math.PI / 5, 0.45, 1.2);
  spot.castShadow = true;
  spot.shadow.mapSize.set(1024, 1024);
  const target = new THREE.Object3D();
  target.position.set(1.5, 0, 0); // lamp head forward = +X
  emitter.add(spot, target);
  spot.target = target;

  const coneLen = 1.1, coneR = Math.tan(Math.PI / 5) * coneLen;
  const coneGeo = new THREE.ConeGeometry(coneR, coneLen, 48, 1, true);
  coneGeo.rotateZ(Math.PI / 2);        // apex toward -X ... then flip:
  coneGeo.translate(coneLen / 2, 0, 0); // apex at emitter, opening along +X
  const coneMat = new THREE.MeshBasicMaterial({
    color: 0xffe9b0, transparent: true, opacity: 0,
    blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
  });
  const cone = new THREE.Mesh(coneGeo, coneMat);
  emitter.add(cone);

  const discs = [];
  for (const g of Object.values(linkGroups)) {
    for (const m of g.userData.emissiveMaterials ?? []) discs.push(m);
  }
  lampLight = { spot, cone, discs, bulb: emitter.userData.bulb ?? null };
}

// ---------- state application (smoothed) ----------
const jointTargets = {};
const lightTarget = { color: new THREE.Color(0xffe9b0), intensity: 0 };
let rootZTarget = 0;

function applyState(msg) {
  Object.assign(jointTargets, msg.joints);
  if (msg.root) rootZTarget = msg.root.z ?? 0;
  if (msg.light) {
    lightTarget.color.setRGB(...msg.light.color);
    lightTarget.intensity = msg.light.intensity;
  }
}

// ---------- viewpoint upstream: tell the character where its audience is ----------
let sock = null;
let _lastAz = null, _lastAzT = 0;
function sendViewpoint() {
  if (!sock || sock.readyState !== WebSocket.OPEN) return;
  // camera azimuth around the lamp in the robot's (URDF, z-up) frame
  const dx = camera.position.x - controls.target.x;
  const dz = camera.position.z - controls.target.z;
  const azimuth = Math.atan2(-dz, dx); // three +Z == URDF -Y
  const now = performance.now();
  if (_lastAz !== null && Math.abs(azimuth - _lastAz) < 0.02 && now - _lastAzT < 500) return;
  _lastAz = azimuth; _lastAzT = now;
  sock.send(JSON.stringify({ type: 'viewpoint', azimuth }));
}

const _q = new THREE.Quaternion();
let lastT = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min((now - lastT) / 1000, 0.1);
  lastT = now;
  const k = 1 - Math.exp(-dt * 24); // light smoothing over 30 Hz stream

  for (const [name, node] of Object.entries(jointNodes)) {
    const target = jointTargets[name];
    if (target === undefined) continue;
    node.current = (node.current ?? 0) + (target - (node.current ?? 0)) * k;
    node.rotor.setRotationFromQuaternion(_q.setFromAxisAngle(node.axis, node.current));
  }
  lampRoot.position.z += (rootZTarget - lampRoot.position.z) * k;
  if (lampLight) {
    const cur = lampLight.spot.intensity;
    const target = lightTarget.intensity * 6; // spot units
    lampLight.spot.intensity = cur + (target - cur) * k;
    lampLight.spot.color.lerp(lightTarget.color, k);
    lampLight.cone.material.color.lerp(lightTarget.color, k);
    lampLight.cone.material.opacity = Math.min(0.16, lightTarget.intensity * 0.16);
    for (const m of lampLight.discs) {
      m.emissive.lerp(lightTarget.color, k);
      m.emissiveIntensity = 0.25 + lightTarget.intensity * 1.6;
    }
    const bulb = lampLight.bulb;
    if (bulb) {
      const i = lightTarget.intensity;
      bulb.glow.intensity = 0.15 + i * 2.0;
      bulb.glow.color.lerp(lightTarget.color, k);
      bulb.filamentMat.color.lerp(lightTarget.color, k);
      bulb.filamentMat.opacity = 0.2 + i * 0.8;
      bulb.glassMat.emissiveIntensity = 0.05 + i * 0.55;
    }
  }
  controls.update();
  sendViewpoint();
  renderer.render(scene, camera);
}
animate();

// ---------- HUD ----------
const stateBadge = document.getElementById('state-badge');
const stateText = document.getElementById('state-text');
const captionBox = document.getElementById('caption');
const captionText = document.getElementById('caption-text');
const captionSpeaker = document.getElementById('caption-speaker');
const conn = document.getElementById('conn');
let captionTimer = null;

function applyHud(msg) {
  if (msg.state) {
    stateText.textContent = msg.state;
    stateBadge.classList.toggle('live', msg.state !== 'idle');
  }
  if (msg.caption !== undefined) {
    if (msg.caption) {
      captionText.textContent = msg.caption;
      captionSpeaker.textContent = msg.speaker ?? '';
      captionBox.classList.add('visible');
      clearTimeout(captionTimer);
      captionTimer = setTimeout(() => captionBox.classList.remove('visible'), 8000);
    } else {
      captionBox.classList.remove('visible');
    }
  }
}

// ---------- device settings panel ----------
const gear = document.getElementById('gear');
const settings = document.getElementById('settings');

async function setDevice(kind, index) {
  await fetch('/devices', { method: 'POST', headers: { 'content-type': 'application/json' },
                            body: JSON.stringify({ [kind]: index }) });
  loadDevices();
}

async function loadDevices() {
  const camsEl = document.getElementById('cams');
  camsEl.textContent = 'scanning…';
  let d;
  try {
    const res = await fetch('/devices');
    if (!res.ok) throw new Error();
    d = await res.json();
  } catch {
    camsEl.textContent = 'unavailable (demo mode)';
    return;
  }
  camsEl.innerHTML = '';
  for (const cam of d.cameras) {
    const div = document.createElement('div');
    div.className = 'cam' + (cam.current ? ' current' : '');
    div.innerHTML = cam.thumb ? `<img src="data:image/jpeg;base64,${cam.thumb}">`
                              : `<img alt="camera ${cam.index}">`;
    div.title = `camera ${cam.index}`;
    div.onclick = () => setDevice('camera', cam.index);
    camsEl.appendChild(div);
  }
  for (const [id, list] of [['mic', d.audio_in], ['spk', d.audio_out]]) {
    const sel = document.getElementById(id);
    sel.innerHTML = '';
    for (const dev of list) {
      const opt = document.createElement('option');
      opt.value = dev.index;
      opt.textContent = dev.name;
      opt.selected = dev.current;
      sel.appendChild(opt);
    }
    sel.onchange = () => setDevice(id === 'mic' ? 'audio_in' : 'audio_out',
                                   parseInt(sel.value, 10));
  }
}

gear.onclick = () => {
  const open = settings.style.display === 'block';
  settings.style.display = open ? 'none' : 'block';
  if (!open) loadDevices();
};

// ---------- websocket ----------
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  sock = ws;
  ws.onopen = () => { conn.textContent = 'connected'; _lastAz = null; sendViewpoint(); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'robot' && !Object.keys(jointNodes).length) buildRobot(msg);
    else if (msg.type === 'state') applyState(msg);
    else if (msg.type === 'hud') applyHud(msg);
    else if (msg.type === 'camera') {
      document.getElementById('pip').style.display = 'block';
      document.getElementById('pip-img').src = `data:image/jpeg;base64,${msg.data}`;
    }
  };
  ws.onclose = () => { conn.textContent = 'offline — retrying'; setTimeout(connect, 1000); };
}
connect();
