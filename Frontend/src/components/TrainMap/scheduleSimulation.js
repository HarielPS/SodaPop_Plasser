// utils/scheduleSimulation.js
import L from "leaflet";

export let globalSimTime = 0;
export function updateSimTime(t) {
  globalSimTime = t;
}

/* -------------------------------------------------------
UTILS
--------------------------------------------------------*/

/** Encuentra el punto más cercano de una ruta */
function findClosestIndex(path, targetLat, targetLon) {
  let best = 0, bestDist = Infinity;
  for (let i = 0; i < path.length; i++) {
    const [lat, lon] = path[i];
    const d = (lat - targetLat) ** 2 + (lon - targetLon) ** 2;
    if (d < bestDist) { bestDist = d; best = i; }
  }
  return best;
}

/** Animación sin duplicar markers */
function animateMarkerAlongPath({
  path,
  durationMs,
  marker,
  registry,
  isTrain,
  lineIdx,
  blockState,
  onStep,
  onComplete,
}) {
  if (!marker || !path || path.length < 2) return;

  const steps = path.length - 1;
  const stepTime = durationMs / steps;
  let idx = 0;

  const safeDistance = 10;

  const intervalId = setInterval(() => {
    if (isTrain && blockState.blocks[lineIdx]) {
      const blockedIdx = blockState.blocks[lineIdx].idx;
      if (idx + safeDistance >= blockedIdx) return;
    }

    idx++;
    if (idx >= path.length) {
      clearInterval(intervalId);
      if (onComplete) onComplete(marker);
      return;
    }

    marker.setLatLng(path[idx]);
    if (onStep) onStep(idx, marker);
  }, stepTime);

  registry.push({ marker, intervalId });
}

/* -------------------------------------------------------
   SIMULADOR PRINCIPAL
--------------------------------------------------------*/
export function scheduleSimulation({
  map,
  layerGroup,
  lineSegments,
  schedule,
  registryRef,
}) {
  const registry = [];
  registryRef.current = registry;

  const blueMachines = {}; // una máquina por ID
  const blockState = { blocks: {} };
  const tampingLaunchedForLine = {};

  /* -------------------------------------------------------
     ICONOS
  --------------------------------------------------------*/
  const trainIcon = L.divIcon({
    html: `<div style="background:#ffcc00;width:24px;height:16px;border-radius:4px;
    box-shadow:0 0 15px #fc0;display:flex;align-items:center;justify-content:center">🚄</div>`,
    iconSize: [24, 16],
    iconAnchor: [12, 8],
  });

  const tampingIcon = L.divIcon({
    html: `<div style="background:#00e0ff;width:22px;height:18px;border-radius:4px;
    box-shadow:0 0 12px #0ef;display:flex;align-items:center;justify-content:center">🛠️</div>`,
    iconSize: [22, 18],
    iconAnchor: [11, 9],
  });

  /* -------------------------------------------------------
     ACTIVAR REPARACIÓN
  --------------------------------------------------------*/
  function triggerTampingForLine(lineIdx) {
    const fullPath = lineSegments[lineIdx];
    if (!fullPath) return;

    const machines = schedule.tampingMachines.filter(
      (m) => m.line - 1 === lineIdx
    );

    machines.forEach((tm) => {
      const targetIndex =
        tm.targetIndex ??
        findClosestIndex(fullPath, tm.targetLat, tm.targetLon);

      const [zLat, zLon] = fullPath[targetIndex];

      blockState.blocks[lineIdx] = { idx: 0 };

      const zoneMarker = L.circleMarker([zLat, zLon], {
        radius: 8,
        color: "#00e0ff",
        fillColor: "#00e0ff",
        fillOpacity: 0.4,
        weight: 2,
      }).addTo(layerGroup);
      registry.push({ marker: zoneMarker });

      L.popup()
        .setLatLng([zLat, zLon])
        .setContent(`<b>⚠ Hundimiento detectado</b><br>Línea ${tm.line}<br>${tm.description}`)
        .openOn(map);

      const pathToTarget = fullPath.slice(0, targetIndex + 1);
      const restPath = fullPath.slice(targetIndex);

      const dwellMs = 4000;
      const travelMs = 3000;

      if (!blueMachines[tm.id]) {
        blueMachines[tm.id] = L.marker(pathToTarget[0], { icon: tampingIcon })
          .addTo(layerGroup);
      }
      const blue = blueMachines[tm.id];

      const onStepTo = (i) => { blockState.blocks[lineIdx] = { idx: i }; };
      const onStepRest = (i) => { blockState.blocks[lineIdx] = { idx: targetIndex + i }; };

      /* --- IR a reparar --- */
      animateMarkerAlongPath({
        path: pathToTarget,
        durationMs: travelMs,
        marker: blue,
        registry,
        isTrain: false,
        lineIdx,
        blockState,
        onStep: onStepTo,
        onComplete: () => {
          setTimeout(() => {
            layerGroup.removeLayer(zoneMarker);
            blockState.blocks[lineIdx] = null;

            /* --- AVANZAR --- */
            animateMarkerAlongPath({
              path: restPath,
              durationMs: 3000,
              marker: blue,
              registry,
              isTrain: false,
              lineIdx,
              blockState,
              onStep: onStepRest,
              onComplete: (marker) => {
                if (marker) layerGroup.removeLayer(marker);
                delete blueMachines[tm.id];
                blockState.blocks[lineIdx] = null;
              },
            });
          }, dwellMs);
        },
      });
    });
  }

  /* -------------------------------------------------------
     TRENES CONTROLADOS POR simTime
  --------------------------------------------------------*/
  function processTrainSchedule() {
    schedule.trains.forEach((train) => {
      const lineIdx = train.line - 1;
      const path = lineSegments[lineIdx];
      if (!path) return;

      if (!train._launched && globalSimTime >= train.departureSec) {
        train._launched = true;

        if (blockState.blocks[lineIdx]) return;

        const marker = L.marker(path[0], { icon: trainIcon }).addTo(layerGroup);

        animateMarkerAlongPath({
          path,
          durationMs: train.travelTimeSec * 1000,
          marker,
          registry,
          isTrain: true,
          lineIdx,
          blockState,
          onComplete: () => {
            layerGroup.removeLayer(marker);

            if (!tampingLaunchedForLine[lineIdx]) {
              tampingLaunchedForLine[lineIdx] = true;
              triggerTampingForLine(lineIdx);
            }
          },
        });
      }
    });
  }

  /* -------------------------------------------------------
     LOOP PRINCIPAL (SIN REINICIO)
  --------------------------------------------------------*/
  setInterval(() => {
    processTrainSchedule();
  }, 200);
}

/* -------------------------------------------------------
   LIMPIEZA
--------------------------------------------------------*/
export function clearSimulation(registryRef, layerGroup) {
  if (registryRef.current) {
  


  registryRef.current.forEach((item) => {
    if (item.intervalId) clearInterval(item.intervalId);
    if (item.marker) layerGroup.removeLayer(item.marker);
  });
  registryRef.current = [];
}
}
