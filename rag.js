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
export async function answerQuestion(question) {
  try {
    if (!vectorStore) return false;

    // await db.read();
    // const chat = db.data.chats.find(c => c.id === req.params.id);

    // chat.messages.push({ id: nanoid(), role:user, text: question });
    // await db.write();

    const retriever = vectorStore.asRetriever(1);
    const retrievedDocuments = await retriever.invoke(question);

    if (!retrievedDocuments || retrievedDocuments.length === 0) {
      return "I couldn't find anything related to your question in the file.";
    }

    const contextText = retrievedDocuments
      .map((r) => r.pageContent)
      .join("\n\n");

    // Google Gemini LLM
    const llm = new ChatGoogleGenerativeAI({
      apiKey: process.env.API_KEY,
      model: "gemini-2.5-flash",
      temperature: 0,
    });

    const prompt = `
Use the following context to answer the question strictly.
If the answer is not in the context, reply: "I couldn't find anything related to your question in the file."

Context:
${contextText}

Question:
${question}
`;

    const answer = await llm.invoke(prompt);
    
    // chat.messages.push({ id: nanoid(), role:bot, text: answer.content });
    // await db.write();
    return answer.content;
  } catch (err) {
    console.error("Error in answerQuestion:", err.message);
    return "Something went wrong while answering the question.";
  }
}
