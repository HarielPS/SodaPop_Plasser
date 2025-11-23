import styles from "./HistoryLog.module.css";
import HistoryRow from "./HistoryRow";

const mockHistory = [
  { id: 1203, action: "Lift correction (12mm)", machine: "Machine 02", time: "14:23" },
  { id: 1194, action: "Alignment fix", machine: "Machine 03", time: "09:12" },
  { id: 1180, action: "Stabilization", machine: "Machine 01", time: "12:55" },
];

export default function HistoryLog() {
  return (
    <div className={styles.box}>
      <h3 className={styles.title}>History Log</h3>

      <table className={styles.table}>
        <thead>
          <tr>
            <th>ID</th>
            <th>Action</th>
            <th>Machine</th>
            <th>Time</th>
          </tr>
        </thead>

        <tbody>
          {mockHistory.map((h) => (
            <HistoryRow key={h.id} row={h} />
          ))}
        </tbody>
      </table>
    </div>
  );
}