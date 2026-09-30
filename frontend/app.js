const API_URL = '/api/chat';

const messages = document.querySelector('#messages');
const form = document.querySelector('#chatForm');
const input = document.querySelector('#messageInput');
const clearButton = document.querySelector('#clearButton');
const suggestionButtons = document.querySelectorAll('[data-prompt]');
let conversationHistory = [];

function scrollToLatest() {
  requestAnimationFrame(() => {
    messages.scrollTo({ top: messages.scrollHeight, behavior: 'smooth' });
  });
}

const allowedTerms = [
  'cs', 'it', 'is', 'cpe', 'bscs', 'bsit', 'bsis', 'bscpe',
  'computer science', 'information technology', 'information systems',
  'computer engineering', 'software engineering', 'programmer', 'coding',
  'developer', 'data science', 'cybersecurity', 'tech', 'computing',
  'career', 'internship', 'resume', 'interview', 'job', 'skills', 'salary',
];

const topics = [
  { intent: 'career_path_exploration', terms: ['career', 'path', 'specialize', 'field', 'direction'] },
  { intent: 'skill_recommendation', terms: ['skill', 'learn', 'study', 'roadmap', 'language', 'tool'] },
  { intent: 'resume_portfolio_guidance', terms: ['resume', 'portfolio', 'github', 'cover letter', 'project'] },
  { intent: 'internship_job_search', terms: ['internship', 'job', 'hire', 'recruiter', 'application', 'ojt'] },
  { intent: 'interview_preparation', terms: ['interview', 'coding questions', 'behavioral', 'system design'] },
  { intent: 'further_studies', terms: ['master', 'phd', 'graduate', 'scholarship', 'grad school'] },
  { intent: 'role_comparison', terms: ['difference', 'compare', 'versus', ' vs ', 'which role'] },
  { intent: 'salary_compensation', terms: ['salary', 'pay', 'earn', 'compensation', 'negotiate'] },
];

const responses = {
  career_path_exploration: 'A CS degree can lead to software engineering, data, cloud, cybersecurity, QA, product, or research roles. Start by comparing the kind of problems you enjoy, then test a path with one small project or internship before committing.',
  skill_recommendation: 'Build a foundation in programming, Git, problem solving, and communication first. Then choose a direction: Python and statistics for data, JavaScript and web fundamentals for product development, or Linux, networking, and cloud basics for infrastructure.',
  resume_portfolio_guidance: 'Keep your resume to one focused page and lead with outcomes, not task lists. Your portfolio should show two or three finished projects with a short explanation of the problem, your contribution, and what you learned.',
  internship_job_search: 'Start early and tailor each application to the role. Use your school network, LinkedIn, and reputable job boards, and keep a simple tracker for applications, follow-ups, and the skills each posting asks for.',
  interview_preparation: 'Practice explaining your thinking out loud. Review data structures and algorithms, prepare three project stories using the situation-action-result format, and finish each practice session by writing down one gap to revisit.',
  further_studies: 'Graduate school makes the most sense when you have a specific research or specialization goal. Compare the cost, curriculum, outcomes, and opportunity cost against gaining practical experience first.',
  role_comparison: 'Compare roles by daily work, not just job titles. Look at the tools used, the type of problems solved, the stakeholders involved, and the kind of work that gives you energy over a normal week.',
  salary_compensation: 'Salary depends on location, role, experience, and company type. Research several current postings, compare the total package rather than base pay alone, and prepare a range supported by your skills and evidence.',
};

const refusal = "I'm sorry, but I can only assist with career questions related to Computer Studies (such as BSCS, BSIT, BSIS, and BSCpE programs).";

function addMessage(text, role, meta = '') {
  const article = document.createElement('article');
  article.className = `message ${role}-message`;
  article.innerHTML = `
    ${role === 'user' ? '<div class="avatar" aria-hidden="true">YOU</div>' : ''}
    <div class="bubble"><p>${escapeHtml(text)}</p>${meta ? `<span class="message-meta">${escapeHtml(meta)}</span>` : ''}</div>
  `;
  messages.append(article);
  scrollToLatest();
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[character]));
}

function detectIntent(message) {
  const normalized = message.toLowerCase();
  const topic = topics
    .map((candidate) => ({ ...candidate, score: candidate.terms.filter((term) => normalized.includes(term)).length }))
    .sort((a, b) => b.score - a.score)[0];
  return topic && topic.score > 0 ? topic : { intent: 'out_of_scope', score: 0 };
}

async function getReply(message) {
  if (API_URL) {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history: conversationHistory }),
    });
    if (!response.ok) throw new Error('CareerBot API request failed');
    const data = await response.json();
    return { text: data.answer || data.response, meta: '' };
  }

  const normalized = message.toLowerCase();
  const hasAllowedTerm = allowedTerms.some((term) => normalized.includes(term));
  const detected = detectIntent(message);
  if (!hasAllowedTerm || detected.intent === 'out_of_scope') {
    return { text: refusal, meta: '' };
  }
  return { text: responses[detected.intent], meta: '' };
}

async function submitMessage(message) {
  const cleanMessage = message.trim();
  if (!cleanMessage) return;
  addMessage(cleanMessage, 'user');
  input.value = '';
  input.style.height = 'auto';
  const typing = document.createElement('article');
  typing.className = 'message bot-message';
  typing.innerHTML = '<div class="bubble"><p>Thinking...</p></div>';
  messages.append(typing);
  scrollToLatest();

  try {
    const reply = await getReply(cleanMessage);
    typing.remove();
    addMessage(reply.text, 'bot', reply.meta);
    conversationHistory.push({ role: 'user', content: cleanMessage });
    conversationHistory.push({ role: 'assistant', content: reply.text });
  } catch (error) {
    typing.remove();
    addMessage('I could not reach the model service right now. Please try again in a moment.', 'bot', 'Connection error');
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  submitMessage(input.value);
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 110)}px`;
});

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

suggestionButtons.forEach((button) => button.addEventListener('click', () => submitMessage(button.dataset.prompt)));

clearButton.addEventListener('click', () => {
  conversationHistory = [];
  messages.innerHTML = '<article class="message bot-message"><div class="bubble"><p>Chat cleared. What would you like to work through?</p><span class="message-meta">CareerBot · ready</span></div></article>';
  scrollToLatest();
  input.focus();
});
