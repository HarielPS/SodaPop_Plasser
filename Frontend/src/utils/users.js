export function getOperators() {
    return JSON.parse(localStorage.getItem("operators")) || [];
  }
  
  export function addOperator(operator) {
    const ops = getOperators();
    ops.push(operator);
    localStorage.setItem("operators", JSON.stringify(ops));
  }