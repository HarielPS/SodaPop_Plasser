import styles from "./MachineCard.module.css";

export default function MachineCard({ title, status, color }) {
  return (
    <div className={styles.card}>
      <h4 className={styles.title}>{title}</h4>
      <p className={styles.status}>
        <span className={styles.dot} style={{ backgroundColor: color }}></span>
        {status}
      </p>
    </div>
  );
}