import styles from "./IssueCard.module.css";

export default function IssueCard({ issue }) {
  return (
    <div className={styles.card}>
      <h4 className={styles.header}>Issue #{issue.id}</h4>

      <p>
        <strong>Severity:</strong> <span className={styles.severity}>{issue.severity}</span>
      </p>

      <p>
        <strong>Description:</strong> {issue.description}
      </p>

      <p>
        <strong>Location:</strong> {issue.location}
      </p>

      <p>
        <strong>Reported:</strong> {issue.date}
      </p>

      <button className={styles.btn}>Assign Machine</button>
    </div>
  );
}