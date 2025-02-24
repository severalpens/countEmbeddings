require('dotenv').config();
var fs = require('fs-extra');
const { Client } = require('@elastic/elasticsearch');
const { pipeline } = require('@xenova/transformers');
const isLocalClient = false;
const node = isLocalClient ? process.env.ELASTICSEARCH_DOCKER_URL : process.env.ELASTICSEARCH_URL;
const apiKey = isLocalClient ? process.env.ELASTIC_DOCKER_API_KEY : process.env.ELASTIC_API_KEY;

const client = new Client({
    node: node,
    auth: {
        apiKey: apiKey
    }
});

const indexName = 'huggingfaceembedding';


const putJson = {
    mappings: {
        properties: {
            text: { type: "text" },
            embedding: { type: "dense_vector", dims: 384 },
            tokenCount: { type: "integer" }
        }
    }
};

let embedder;
async function loadEmbedder() {
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
}

async function ensureIndexExists() {
    const indexExists = await client.indices.exists({ index: indexName });
    if (indexExists) {
        console.log(`Index ${indexName} already exists`);
        console.log(`Deleting index: ${indexName} `);
        await client.indices.delete({ index: indexName });
    }

    await client.indices.create({
        index: indexName,
        body: putJson
    });
}

const embed = async (text) => {
    const embeddingTensor = await embedder(text, { pooling: 'mean', normalize: true });
    const embedding = Array.from(embeddingTensor.data); // Convert tensor to plain array
    return embedding;
};

async function main() {
    await loadEmbedder();
    await ensureIndexExists();
    var faqs = fs.readJsonSync('faqs.json');
    for (const faq of faqs) {
        const document = {
            text: faq.Answer,
            embedding: await embed(faq.Answer),
            tokenCount: 0
        };

        const result = await client.index({
            index: indexName,
            body: document
        });

        console.log("result:", result);
    }
}

main().catch(console.error);

