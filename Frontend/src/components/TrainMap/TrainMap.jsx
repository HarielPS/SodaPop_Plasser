import styles from "./TrainMap.module.css";

export default function TrainMap() {
  return (
    <div className={styles.mapContainer}>
      <h3 className={styles.title}>Railway Map</h3>

      <div className={styles.map}>
        <div className={styles.track}></div>
        <div className={styles.train}></div>
      </div>
    </div>
  );
}