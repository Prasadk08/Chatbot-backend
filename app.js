import express from "express"
import cors from "cors"
import dotenv from "dotenv"
import multer from "multer"
// import { Low } from "lowdb";
// import { JSONFile } from 'lowdb/node'
import { answerQuestion, processFileAndStoreVectors } from "./rag.js"
const app = express()

app.use(
  cors({
    origin: ["http://localhost:5173", "https://chatbot-frontend-vlwr.vercel.app"],
    credentials: true,
  })
);


dotenv.config()
app.use(express.json())

// // Create JSON DB
// const adapter = new JSONFile("db.json")
// export const db = new Low(adapter, { chats: [] })

// // Initialize DB
// await db.read();
// db.data = db.data || { chats: [] };
// await db.write();


const port= 8080

const upload = multer({dest:"uploads/"})

app.post("/upload",upload.single("file"),async(req,res)=>{
 
    try{
        const file = req.file.path
        await processFileAndStoreVectors(file)
        res.json({ message: "File processed successfully"});
    }catch(e){
        console.log(e)
        res.status(500).json({error:"Something Went Wrong"})
    }
    

})

app.post("/ask",async(req,res)=>{
   try {
    const { question, history } = req.body;

    const answer = await answerQuestion(question, history);

    if (answer) {
      res.status(201).json({ answer });
      return;
    }

    res.status(202).json({ message: "Upload the File First" });
  } catch (e) {
    console.log(e);
    res.status(500).json({ error: "Something Went Wrong" });
  }

})

app.listen(port,(req,res)=>{
    console.log(`Server is running on Port ${port}`)
})