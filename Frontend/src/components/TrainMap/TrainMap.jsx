import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import styles from "./TrainMap.module.css";

import { drawRailLines } from "./drawRailLines";
import { scheduleSimulation, clearSimulation } from "./scheduleSimulation";

export default function TrainMap() {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const routeLayerRef = useRef(null);

  // Para poder limpiar animaciones (trenes + tamping)
  const simulationRegistryRef = useRef(null);

  useEffect(() => {
    // Crear mapa solo una vez
    if (!mapInstanceRef.current && mapRef.current) {
      mapInstanceRef.current = L.map(mapRef.current, {
        center: [45.0, 10.0],
        zoom: 5,
        zoomControl: true,
      });

      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        { maxZoom: 19 }
      ).addTo(mapInstanceRef.current);
    }

    // Crear / limpiar capa de rutas
    if (routeLayerRef.current) {
      mapInstanceRef.current.removeLayer(routeLayerRef.current);
    }
    routeLayerRef.current = L.layerGroup().addTo(mapInstanceRef.current);

    // ============ LEER CSV DE VÍAS ============
    fetch("/resources/datos_inspeccion_vias.csv")
      .then((res) => res.text())
      .then((text) => {
        const rows = text.trim().split("\n");
        const header = rows[0].split(",");

        const latIdx = header.indexOf("Lat_caro");
        const lonIdx = header.indexOf("Lon_caro");

        if (latIdx === -1 || lonIdx === -1) {
          console.error("No existe Lat_caro / Lon_caro en el CSV");
          return;
        }

        const latlngs = [];
        for (let i = 1; i < rows.length; i++) {
          const cols = rows[i].split(",");
          const lat = parseFloat(cols[latIdx]);
          const lon = parseFloat(cols[lonIdx]);
          if (!isNaN(lat) && !isNaN(lon)) latlngs.push([lat, lon]);
        }

        if (latlngs.length === 0) return;

        // Icono de estación
        const stationIcon = L.divIcon({
          html: `<div style="
            background:white;
            width:10px;
            height:10px;
            border-radius:50%;
            border:3px solid #ffcc00;
            box-shadow:0 0 8px rgba(255,204,0,0.8);
          "></div>`,
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        });

        // ----- Dibujar las 3 líneas y obtener segmentos + bounds
        const { lineSegments, bounds } = drawRailLines({
          map: mapInstanceRef.current,
          layerGroup: routeLayerRef.current,
          latlngs,
          stationIcon,
        });

        if (bounds.isValid()) {
          mapInstanceRef.current.fitBounds(bounds, { padding: [40, 40] });
        }

        // ============ LEER JSON DE HORARIOS Y SIMULAR ============
        fetch("/resources/trains_schedule.json")
          .then((r) => r.json())
          .then((schedule) => {
            scheduleSimulation({
              map: mapInstanceRef.current,
              layerGroup: routeLayerRef.current,
              lineSegments,
              schedule,
              registryRef: simulationRegistryRef,
            });
          })
          .catch((err) =>
            console.error("Error cargando trains_schedule.json", err)
          );
      })
      .catch((err) =>
        console.error("Error cargando datos_inspeccion_vias.csv", err)
      );

    // Cleanup al desmontar
    return () => {
      if (routeLayerRef.current) {
        clearSimulation(simulationRegistryRef, routeLayerRef.current);
      }
    };
  }, []);

  return <div ref={mapRef} className={styles.map}></div>;
}
