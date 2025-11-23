import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import styles from "./TrainMap.module.css";

export default function TrainMap() {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const routeLayerRef = useRef(null);

  // Coordenadas de ejemplo (puedes conectar esto a tu estado global o props)
  // Ciudad de México a Querétaro
  const [startLat] = useState(19.4326);
  const [startLng] = useState(-99.1332);
  const [endLat] = useState(20.5888);
  const [endLng] = useState(-100.3899);

  useEffect(() => {
    // Inicializar el mapa solo una vez
    if (!mapInstanceRef.current && mapRef.current) {
      mapInstanceRef.current = L.map(mapRef.current, {
        center: [19.4326, -99.1332],
        zoom: 7,
        zoomControl: true,
        attributionControl: false, // Ocultar atribución para más espacio
      });

      // Añadir capa de mapa oscuro
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '',
        maxZoom: 19,
      }).addTo(mapInstanceRef.current);
    }

    // Actualizar la ruta cuando cambien las coordenadas
    if (mapInstanceRef.current && startLat && startLng && endLat && endLng) {
      // Remover la ruta anterior si existe
      if (routeLayerRef.current) {
        mapInstanceRef.current.removeLayer(routeLayerRef.current);
      }

      // Crear grupo de capas para la ruta
      routeLayerRef.current = L.layerGroup().addTo(mapInstanceRef.current);

      // Icono personalizado para estaciones
      const stationIcon = L.divIcon({
        className: 'custom-station-icon',
        html: `<div style="
          background: #ffcc00;
          width: 14px;
          height: 14px;
          border-radius: 50%;
          border: 3px solid #fff;
          box-shadow: 0 0 12px rgba(255, 204, 0, 0.9);
        "></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      });

      // Añadir marcadores de inicio y fin
      L.marker([startLat, startLng], { icon: stationIcon })
        .bindPopup('<div style="color: #ffcc00; font-weight: 600;">🚉 Estación Origen</div>')
        .addTo(routeLayerRef.current);

      L.marker([endLat, endLng], { icon: stationIcon })
        .bindPopup('<div style="color: #ffcc00; font-weight: 600;">🚉 Estación Destino</div>')
        .addTo(routeLayerRef.current);

      // Crear la línea ferroviaria principal
      L.polyline(
        [[startLat, startLng], [endLat, endLng]],
        {
          color: '#ffcc00',
          weight: 5,
          opacity: 0.9,
          dashArray: '12, 8',
        }
      ).addTo(routeLayerRef.current);

      // Línea paralela para simular segunda vía
      const offset = 0.008;
      L.polyline(
        [[startLat + offset, startLng], [endLat + offset, endLng]],
        {
          color: '#ffcc00',
          weight: 5,
          opacity: 0.9,
          dashArray: '12, 8',
        }
      ).addTo(routeLayerRef.current);

      // Añadir icono de tren en movimiento
      const trainIcon = L.divIcon({
        className: 'train-icon',
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

      // Posición del tren (30% del camino)
      const trainLat = startLat + (endLat - startLat) * 0.3;
      const trainLng = startLng + (endLng - startLng) * 0.3;
      
      L.marker([trainLat, trainLng], { icon: trainIcon })
        .bindPopup('<div style="color: #ffcc00; font-weight: 600;">🚄 Tren en tránsito</div>')
        .addTo(routeLayerRef.current);

      // Calcular el centro de la ruta y ajustar la vista
      const bounds = L.latLngBounds([
        [startLat, startLng],
        [endLat, endLng]
      ]);
      mapInstanceRef.current.fitBounds(bounds, { padding: [30, 30] });
    }

    return () => {
      // Cleanup de capas, no del mapa completo
    };
  }, [startLat, startLng, endLat, endLng]);

  // Cleanup al desmontar el componente
  useEffect(() => {
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  return (
    <div className={styles.mapContainer}>
      <h3 className={styles.title}>Railway Map</h3>
      <div ref={mapRef} className={styles.map}></div>
      <div className={styles.routeInfo}>
        <div className={styles.routeDetail}>
          <span className={styles.label}>Origen:</span>
          <span className={styles.value}>Ciudad de México</span>
        </div>
        <div className={styles.routeDetail}>
          <span className={styles.label}>Destino:</span>
          <span className={styles.value}>Querétaro</span>
        </div>
      </div>
    </div>
  );
}