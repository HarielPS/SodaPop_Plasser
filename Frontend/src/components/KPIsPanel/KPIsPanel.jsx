import styles from "./KPIsPanel.module.css";

export default function KPIsPanel() {
  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <h4>System Uptime</h4>
        <p className={styles.value}>99.7%</p>
      </div>

      <div className={styles.card}>
        <h4>GNSS Accuracy</h4>
        <p className={styles.value}>1.3 m</p>
      </div>

      <div className={styles.card}>
        <h4>Alerts Today</h4>
        <p className={styles.value}>4</p>
      </div>

      <div className={styles.card}>
        <h4>Last Lift Applied</h4>
        <p className={styles.value}>2h ago</p>
      </div>
    </div>
  );
}