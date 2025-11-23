import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import styles from "./TrainMap.module.css";

export default function TrainMap({ railwayData }) {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const routeLayerRef = useRef(null);
  const [selectedSection, setSelectedSection] = useState(null);
  const [stats, setStats] = useState({ total: 0, critical: 0, high: 0, medium: 0, low: 0 });

  useEffect(() => {
    // Inicializar el mapa solo una vez
    if (!mapInstanceRef.current && mapRef.current) {
      mapInstanceRef.current = L.map(mapRef.current, {
        center: [45.0, 10.0], // Centro por defecto
        zoom: 5,
        zoomControl: true,
        attributionControl: false,
      });

      // Añadir capa de mapa oscuro
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '',
        maxZoom: 19,
      }).addTo(mapInstanceRef.current);
    }

    // Dibujar rutas cuando cambien los datos
    if (mapInstanceRef.current && railwayData && railwayData.length > 0) {
      // Remover las rutas anteriores si existen
      if (routeLayerRef.current) {
        mapInstanceRef.current.removeLayer(routeLayerRef.current);
      }

      // Crear grupo de capas para las rutas
      routeLayerRef.current = L.layerGroup().addTo(mapInstanceRef.current);

      // Array para calcular los límites del mapa
      const allCoordinates = [];
      
      // Estadísticas
      let critical = 0, high = 0, medium = 0, low = 0;

      // Función para determinar color basado en el hundimiento
      const getColorByHundimiento = (hundimiento) => {
        if (hundimiento > 1500000) {
          critical++;
          return '#ff0000'; // Rojo - crítico
        }
        if (hundimiento > 1000000) {
          high++;
          return '#ff8800'; // Naranja - alto
        }
        if (hundimiento > 500000) {
          medium++;
          return '#ffcc00'; // Amarillo - medio
        }
        low++;
        return '#88ff00'; // Verde-amarillo - bajo
      };

      // Dibujar cada sección de la ruta
      railwayData.forEach((section) => {
        const {
          seccion_id,
          lat_inicio,
          lon_inicio,
          lat_fin,
          lon_fin,
          hundimiento_mm,
          ajuste_izquierdo_mm,
          ajuste_derecho_mm
        } = section;

        // Añadir coordenadas al array para calcular límites
        allCoordinates.push([lat_inicio, lon_inicio]);
        allCoordinates.push([lat_fin, lon_fin]);

        const lineColor = getColorByHundimiento(hundimiento_mm);

        // Crear la línea de la sección
        const routeLine = L.polyline(
          [[lat_inicio, lon_inicio], [lat_fin, lon_fin]],
          {
            color: lineColor,
            weight: 3,
            opacity: 0.8,
          }
        );

        // Popup con información de la sección
        const popupContent = `
          <div style="color: #fff; font-family: sans-serif; min-width: 200px;">
            <div style="font-weight: 600; font-size: 14px; color: ${lineColor}; margin-bottom: 8px;">
              🚂 Sección ${seccion_id}
            </div>
            <div style="font-size: 12px; line-height: 1.6;">
              <div><strong>Hundimiento:</strong> ${(hundimiento_mm / 1000).toFixed(2)} m</div>
              <div><strong>Ajuste Izq:</strong> ${ajuste_izquierdo_mm} mm</div>
              <div><strong>Ajuste Der:</strong> ${ajuste_derecho_mm} mm</div>
              <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #444;">
                <strong>Inicio:</strong> ${lat_inicio.toFixed(4)}, ${lon_inicio.toFixed(4)}<br>
                <strong>Fin:</strong> ${lat_fin.toFixed(4)}, ${lon_fin.toFixed(4)}
              </div>
            </div>
          </div>
        `;

        routeLine.bindPopup(popupContent);
        
        // Evento de click para seleccionar sección
        routeLine.on('click', () => {
          setSelectedSection(section);
        });

        routeLine.addTo(routeLayerRef.current);

        // Añadir marcadores pequeños en los puntos si inicio y fin son diferentes
        if (lat_inicio !== lat_fin || lon_inicio !== lon_fin) {
          const pointIcon = L.divIcon({
            className: 'section-point',
            html: `<div style="
              background: ${lineColor};
              width: 6px;
              height: 6px;
              border-radius: 50%;
              border: 1px solid #fff;
              box-shadow: 0 0 4px rgba(0,0,0,0.5);
            "></div>`,
            iconSize: [6, 6],
            iconAnchor: [3, 3],
          });

          L.marker([lat_inicio, lon_inicio], { icon: pointIcon })
            .addTo(routeLayerRef.current);
        }
      });

      // Actualizar estadísticas
      setStats({
        total: railwayData.length,
        critical,
        high,
        medium,
        low
      });

      // Ajustar vista del mapa para mostrar todas las rutas
      if (allCoordinates.length > 0) {
        const bounds = L.latLngBounds(allCoordinates);
        mapInstanceRef.current.fitBounds(bounds, { padding: [20, 20] });
      }
    }

    return () => {
      // Cleanup de capas, no del mapa completo
    };
  }, [railwayData]);

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
      <div className={styles.header}>
        <h3 className={styles.title}>Railway Map</h3>
        <div className={styles.sectionCount}>
          {stats.total} Secciones
        </div>
      </div>
      
      <div ref={mapRef} className={styles.map}></div>
      
      <div className={styles.legend}>
        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: '#ff0000' }}></span>
          <span className={styles.legendLabel}>Crítico ({stats.critical})</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: '#ff8800' }}></span>
          <span className={styles.legendLabel}>Alto ({stats.high})</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: '#ffcc00' }}></span>
          <span className={styles.legendLabel}>Medio ({stats.medium})</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: '#88ff00' }}></span>
          <span className={styles.legendLabel}>Bajo ({stats.low})</span>
        </div>
      </div>

      {selectedSection && (
        <div className={styles.selectedInfo}>
          <div className={styles.selectedHeader}>
            <span>Sección {selectedSection.seccion_id} seleccionada</span>
            <button 
              className={styles.closeBtn}
              onClick={() => setSelectedSection(null)}
            >
              ✕
            </button>
          </div>
          <div className={styles.selectedData}>
            <div>Hundimiento: {(selectedSection.hundimiento_mm / 1000).toFixed(2)} m</div>
            <div>Ajustes: {selectedSection.ajuste_izquierdo_mm} mm / {selectedSection.ajuste_derecho_mm} mm</div>
          </div>
        </div>
      )}
    </div>
  );
}