import styles from "./AlertsPanel.module.css";

export default function AlertsPanel() {
  return (
    <div className={styles.panel}>
      <h3 className={styles.title}>Active Alerts</h3>

      <div className={styles.alert}>
        <h4>Lift Alert #1203</h4>
        <p>Vertical lift required: 12 mm</p>
        <button className={styles.btn}>Assign Machine</button>
      </div>
    </div>
  );
}