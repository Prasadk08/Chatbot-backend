import fs from "fs";
import dotenv from "dotenv";
dotenv.config();

// PDF / Text loaders
// import { PDFLoader, TextLoader } from "langchain/document_loaders";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
// import { TextLoader } from "langchain/document_loaders/fs/text";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
// import { MemoryVectorStore } from "@langchain/community/vectorstores/memory";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
// import { nanoid } from "nanoid";

import {
  GoogleGenerativeAIEmbeddings,
  ChatGoogleGenerativeAI,
} from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
// import { db } from "./app.js";

let vectorStore;


// 1️⃣ Process file and store vectors
export async function processFileAndStoreVectors(file) {
  try {

    const loader = new PDFLoader(file);

    const docs = await loader.load();

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const allSplits = await textSplitter.splitDocuments(docs);

    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: "gemini-embedding-001", // 768 dimensions
      taskType: TaskType.RETRIEVAL_DOCUMENT,
      apiKey:process.env.API_KEY,
      title: "Document title",
    });

    vectorStore = await MemoryVectorStore.fromDocuments(allSplits, embeddings);
    // const newChat = { id: nanoid(), title:"New Chat", messages: [] };
    // db.data.chats.push(newChat);
    // await db.write();
    // return newChat
    // console.log("This is the data in vectorStore ",vectorStore)
    // console.log("File processed and vectors stored in memory!");
  } catch (err) {
    console.error("Error in processFileAndStoreVectors:", err.message);
    throw err;
  }
}

// 2️⃣ Answer user question using RAG
export async function answerQuestion(question,history=[]) {
  try {
    if (!vectorStore) return false;

    // Retrieve relevant PDF content
    const retriever = vectorStore.asRetriever(2);
    const retrievedDocuments = await retriever.invoke(question);

    let contextText = "";
    if (retrievedDocuments && retrievedDocuments.length > 0) {
      contextText = retrievedDocuments.map((r) => r.pageContent).join("\n\n");
    }

    // Convert chat history into readable format
    const chatContext = history
      .map((msg) => `${msg.role === "user" ? "User" : "Bot"}: ${msg.text}`)
      .join("\n");

    // Create prompt with both chat history + PDF content
    const prompt = `
You are a helpful assistant answering based on both the conversation history and PDF context.

If the user's question refers to something mentioned earlier in the chat (like "previous question"), use the chat history to answer.

If the question refers to content in the uploaded file, use the PDF context.

If neither contains relevant info, reply: "I couldn't find anything related to your question in the file."

---
Chat History:
${chatContext || "No previous chat history"}

PDF Context:
${contextText || "No context found"}

User Question:
${question}
`;

    const llm = new ChatGoogleGenerativeAI({
      apiKey: process.env.API_KEY,
      model: "gemini-2.5-flash",
      temperature: 0.2,
    });

    const answer = await llm.invoke(prompt);
    return answer.content;

  } catch (err) {
    console.error("Error in answerQuestion:", err.message);
    return "Something went wrong while answering the question.";
  }
}
