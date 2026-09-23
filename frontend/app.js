const uploadForm = document.querySelector('#upload-form');
const questionForm = document.querySelector('#question-form');
const uploadStatus = document.querySelector('#upload-status');
const questionStatus = document.querySelector('#question-status');
const answerBox = document.querySelector('#answer');
const documentSelect = document.querySelector('#document-select');

let conversationHistory = [];``

async function readResponse(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Request failed.');
  return data;
}

async function refreshDocumentList() {
  try {
    const documents = await readResponse(await fetch('/api/documents'));
    const currentValue = documentSelect.value;
    documentSelect.innerHTML = '<option value="">All documents</option>' + documents
      .map((document) => `<option value="${document.document_id}">${document.filename}</option>`)
      .join('');
    if (documents.some((document) => document.document_id === currentValue)) {
      documentSelect.value = currentValue;
    }
  } catch (error) {
    documentSelect.innerHTML = '<option value="">All documents</option>';
  }
}

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  uploadStatus.className = 'status';
  uploadStatus.textContent = 'Processing PDF...';
  const file = document.querySelector('#pdf').files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append('file', file);
  try {
    const data = await readResponse(await fetch('/api/documents', { method: 'POST', body: formData }));
    uploadStatus.textContent = `${data.filename}: ${data.pages_processed} pages and ${data.chunks_created} chunks ready.`;
    await refreshDocumentList();
  } catch (error) {
    uploadStatus.className = 'status error';
    uploadStatus.textContent = error.message;
  }
});

questionForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  questionStatus.className = 'status';
  questionStatus.textContent = 'Searching and generating...';
  answerBox.className = 'answer hidden';
  answerBox.textContent = '';
  try {
    const payload = { question: document.querySelector('#question').value, history: conversationHistory };
    if (documentSelect.value) {
      payload.document_id = documentSelect.value;
    }
    const data = await readResponse(await fetch('/api/questions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }));
    questionStatus.textContent = data.grounded ? 'Answer grounded in retrieved passages.' : 'No supporting passage was found.';
    answerBox.className = 'answer';
    answerBox.textContent = data.answer;
    conversationHistory.push(
      { role: 'user', content: payload.question },
      { role: 'assistant', content: data.answer }
    );
    if (conversationHistory.length > 10)
      {
      conversationHistory = conversationHistory.slice(-10);
    }

    if (data.sources.length) {
      const sources = document.createElement('div');
      sources.className = 'sources';
      sources.textContent = `Sources: ${data.sources.map((source) => `${source.filename}, page ${source.page_number}`).join(' | ')}`;
      answerBox.appendChild(sources);
    }
  } catch (error) {
    questionStatus.className = 'status error';
    questionStatus.textContent = error.message;
  }
});

refreshDocumentList();
