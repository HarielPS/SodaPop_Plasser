import styles from "./ArrivalTimer.module.css";

export default function ArrivalTimer() {
  return (
    <div className={styles.box}>
      <h3 className={styles.title}>Next Train</h3>
      <p className={styles.time}>04:23</p>
      <span className={styles.status}>On Schedule</span>
    </div>
  );
}
