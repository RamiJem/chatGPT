import { useEffect, useState } from "react";

export default  function Index() {

  const [output, setOutput] = useState("")

  useEffect(() => {
    async function getData() {
      // const data = {"hi": "hey"}
      const url = "https://wild-grass-7851.fly.dev/"
      const local_url = "http://127.0.0.1:8000"
      const res = await fetch(url, {
        method: 'POST', // *GET, POST, PUT, DELETE, etc.
        headers: {
          'Content-Type': 'application/json'
        },

        // body: JSON.stringify(data) // body data type must match "Content-Type" header
      });

      const jsonValue = await res.json()
      // const resolved = await Promise.resolve(jsonValue)
      console.log(jsonValue)
      setOutput(jsonValue.message)

    }
    getData()
  }, [])
  
  return (
    <div style={{ fontFamily: "system-ui, sans-serif", lineHeight: "1.4" }}>
      <h1>Character-level decoder only GPT</h1>
      <h2>Trained on <a href="https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt">tiny sheakespeare</a> with a single CPU, so will probably not produce the new Hamlet.</h2>
      <h2 style={{background: "#FFC0CB"}}>{output}</h2>
      <h2>Sometimes it almost sounds like proper English!</h2>
    </div>
  );
}
