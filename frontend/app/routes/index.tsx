import { useEffect, useState } from "react";

export default  function Index() {

  const [output, setOutput] = useState("")

  useEffect(() => {
    async function getData() {
      // const data = {"hi": "hey"}
      const res = await fetch("https://wild-grass-7851.fly.dev/", {
        method: 'POST', // *GET, POST, PUT, DELETE, etc.
        headers: {
          'Content-Type': 'application/json'
        },

        // body: JSON.stringify(data) // body data type must match "Content-Type" header
      });

      const jsonValue = await res.json()
      const resolved = await Promise.resolve(jsonValue)
      console.log(jsonValue)
      console.log(resolved)
      setOutput(jsonValue.message)
      // console.log("something")
    }
    getData()
  }, [])
  
  return (
    <div style={{ fontFamily: "system-ui, sans-serif", lineHeight: "1.4" }}>
      <h1>Welcome to Remix</h1>
      <h2>Output is: {output}</h2>
    </div>
  );
}
