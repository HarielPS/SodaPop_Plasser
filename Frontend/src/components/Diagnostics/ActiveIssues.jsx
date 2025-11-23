import styles from "./ActiveIssues.module.css";
import IssueCard from "./IssueCard";

const mockIssues = [
  {
    id: "1203",
    severity: "High",
    description: "Vertical lift required: 12 mm",
    location: "19.4326, -99.1332",
    date: "14:23",
  },
  {
    id: "1188",
    severity: "Medium",
    description: "Alignment deviation detected",
    location: "19.4371, -99.1290",
    date: "13:10",
  },
];

export default function ActiveIssues() {
  return (
    <div className={styles.box}>
      <h3 className={styles.title}>Active Issues</h3>

      {mockIssues.map((issue) => (
        <IssueCard key={issue.id} issue={issue} />
      ))}
    </div>
  );
}