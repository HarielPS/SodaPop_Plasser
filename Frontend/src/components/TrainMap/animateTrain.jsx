// utils/animateTrain.js
import L from "leaflet";

export function animateTrainOnPath({
  map,
  layerGroup,
  path,
  totalDuration = 30000, // duración total del recorrido (30s)
  trainMarkerRef,
  trainAnimationRef,
}) {
  if (!map || !layerGroup || !path || path.length < 2) {
    console.warn("No se puede animar el tren: ruta inválida");
    return;
  }

  // Limpiar animación previa
  if (trainAnimationRef.current) {
    clearInterval(trainAnimationRef.current);
    trainAnimationRef.current = null;
  }

  if (trainMarkerRef.current) {
    map.removeLayer(trainMarkerRef.current);
    trainMarkerRef.current = null;
  }

  // Icono del tren
  const trainIcon = L.divIcon({
    html: `<div style="
      background: #ffcc00;
      width: 24px;
      height: 16px;
      border-radius: 4px;
      box-shadow: 0 0 15px rgba(255, 204, 0, 0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
    ">🚄</div>`,
    iconSize: [24, 16],
    iconAnchor: [12, 8],
  });

  // Crear marker de tren en el inicio
  const marker = L.marker(path[0], { icon: trainIcon }).addTo(layerGroup);
  trainMarkerRef.current = marker;

  let currentStep = 0;
  const steps = path.length - 1;
  const stepTime = totalDuration / steps;

  const intervalId = setInterval(() => {
    currentStep++;

    if (currentStep >= path.length) {
      clearInterval(intervalId);
      trainAnimationRef.current = null;
      return;
    }

    marker.setLatLng(path[currentStep]);
  }, stepTime);

  trainAnimationRef.current = intervalId;
}
