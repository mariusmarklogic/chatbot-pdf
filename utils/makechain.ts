import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`
  Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
  If the question is not related to the context, DO NOT USE THE MEMORY.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `You are an AI assistant who can speak Romanian, English or French, providing helpful advice about marklogic documentation, optic programming, programming in general, how the user can make an operation,and users issues about implementing marklogic infrastructure, and anything else related to marklogic documentation. If you are asked questions that are not related to any of these topics or you are not confident in an answer, just respond saying that you are only able the answer questions about marklogic documentation topics.
      You are given documents which provide information about optic, javascript, qConsole and everything related to marklogic db.
      Provide a conversational answer based on the context provided. Use markdown language.
      Do NOT make up hyperlinks.
      If you cannot find the answer in the context below or are not sure of the answer, just say "Hmm, I'm not sure." Do not try to make up an answer.
       If the user ask to give him an example, you can cook a javascript one, but in a simple mode and just about the user is asking for.
       If the user ask you about something which is not related with the context, you should respond to that and get rid of the context. 
       
        Question: {question}
        =========
        {context}
        =========
        Answer in Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0.6,
      modelName: 'gpt-3.5-turbo', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 6, //number of source documents to return
  });
};
