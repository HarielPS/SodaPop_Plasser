import styles from "./MachinesStatus.module.css";
import MachineCard from "./MachineCard";

export default function MachinesStatus() {
  return (
    <div className={styles.container}>
      <MachineCard title="Machine 1" status="Available" color="green" />
      <MachineCard title="Machine 2" status="Busy (Resolving Alert)" color="red" />
      <MachineCard title="Machine 3" status="Idle" color="yellow" />
    </div>
  );
}