require('dotenv').config();
var fs = require('fs-extra');
const { Client } = require('@elastic/elasticsearch');
const { pipeline, AutoTokenizer } = require('@xenova/transformers');
const isLocalClient = false;
const node = isLocalClient ? process.env.ELASTICSEARCH_DOCKER_URL : process.env.ELASTICSEARCH_URL;
const apiKey = isLocalClient ? process.env.ELASTIC_DOCKER_API_KEY : process.env.ELASTIC_API_KEY;

const client = new Client({
    node: node,
    auth: {
        apiKey: apiKey
    }
});

const indexName = 'test2';


const putJson = {
    mappings: {
        properties: {
            text: { type: "text" },
            embedding: { type: "dense_vector", dims: 384 },
            tokencount: { type: "integer" }
        }
    }
};


let embedder, tokenizer;

async function loadEmbedder() {
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    tokenizer = await AutoTokenizer.from_pretrained('Xenova/all-MiniLM-L6-v2');
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
    const tokens = await tokenizer.encode(text);
    const tokenCount = tokens.length; // Get token count from tokenizer
    console.log('tokenCount:', tokenCount);

    const embeddingTensor = await embedder(text, { pooling: 'mean', normalize: true });
    const embedding = Array.from(embeddingTensor.data); // Convert tensor to array

    return { embedding, tokenCount };
};
async function main() {
    await loadEmbedder();
    await ensureIndexExists();
    var faqs = fs.readJsonSync('faqs.json');
    var results = [];
    var faqsOver512 = [];
    var i = 0;
    var numberOfDocsWithTokenLengthGTE512 = 0;
    for (const faq of faqs) {
        i++;
        const { embedding, tokenCount } = await embed(faq.Answer);

        const document = {
            text: faq.Answer,
            embedding: embedding,
            tokencount: tokenCount
        };

        results.push({id: i, tokenCount: tokenCount, embeddingCount: embedding.length});

        if(tokenCount >= 512){
            numberOfDocsWithTokenLengthGTE512++;
            faqsOver512.push(faq);
        }

        const result = await client.index({
            index: indexName,
            body: document
        });
    }

    
    fs.writeFileSync('results.csv', 'id|tokenCount\n');

    results.forEach(result => {
        fs.appendFileSync('results.csv', `${result.id}|${result.tokenCount}\n`);
    });

    fs.writeJsonSync('faqsOver512.json', faqsOver512);

    console.log(`${numberOfDocsWithTokenLengthGTE512}/${i}`);
}

main().catch(console.error);

