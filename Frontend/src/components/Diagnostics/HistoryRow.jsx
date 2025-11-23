export default function HistoryRow({ row }) {
    return (
      <tr>
        <td>{row.id}</td>
        <td>{row.action}</td>
        <td>{row.machine}</td>
        <td>{row.time}</td>
      </tr>
    );
  }