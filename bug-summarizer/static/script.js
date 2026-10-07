/**
 * script.js — Client-side logic for Python Bug Summarizer
 * Handles: code analysis, example loading, multi-issue UI rendering
 */

// ============================================================
// Example snippets (for "Try an example" buttons)
// ============================================================
const EXAMPLES = [
    {
        label: "Off-by-one Error",
        code: `def calculate_sum(numbers):
    """Sum all elements in the list"""
    total = 0
    for i in range(len(numbers) + 1):
        total += numbers[i]
    return total`
    },
    {
        label: "Missing Return",
        code: `def find_maximum(values):
    """Find the maximum value in a list"""
    best = values[0]
    for item in values:
        if item > best:
            best = item`
    },
    {
        label: "Undefined Variable",
        code: `def compute_average(data):
    total = sum(data)
    count = len(data)
    return averge / count`
    },
    {
        label: "Wrong Operator",
        code: `def get_average(scores):
    """Calculate the average score"""
    total = sum(scores)
    return total * len(scores)`
    },
    {
        label: "Clean Code",
        code: `def binary_search(items, target):
    """Search for target in sorted list"""
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        elif items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1`
    },
];

// ============================================================
// DOM elements
// ============================================================
const codeInput = document.getElementById("codeInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const charCount = document.getElementById("charCount");
const resultSection = document.getElementById("resultSection");
const errorSection = document.getElementById("errorSection");

// ============================================================
// Character counter
// ============================================================
codeInput.addEventListener("input", () => {
    const len = codeInput.value.length;
    charCount.textContent = `${len.toLocaleString()} / 10,000`;
    charCount.style.color = len > 9000 ? "#ef4444" : "";
});

// Allow Tab key in textarea
codeInput.addEventListener("keydown", (e) => {
    if (e.key === "Tab") {
        e.preventDefault();
        const start = codeInput.selectionStart;
        const end = codeInput.selectionEnd;
        codeInput.value =
            codeInput.value.substring(0, start) +
            "    " +
            codeInput.value.substring(end);
        codeInput.selectionStart = codeInput.selectionEnd = start + 4;
    }
});

// ============================================================
// Load example code
// ============================================================
function loadExample(index) {
    const example = EXAMPLES[index];
    if (!example) return;
    codeInput.value = example.code;
    codeInput.dispatchEvent(new Event("input"));
    // Hide previous results
    resultSection.style.display = "none";
    errorSection.style.display = "none";
    // Scroll to input
    codeInput.scrollIntoView({ behavior: "smooth", block: "center" });
    codeInput.focus();
}

// ============================================================
// Analyze code (call /predict)
// ============================================================
async function analyzeCode() {
    const code = codeInput.value.trim();

    // Validate
    if (!code) {
        showError("Please paste some Python code before analyzing.");
        return;
    }

    // Toggle loading state
    setLoading(true);
    resultSection.style.display = "none";
    errorSection.style.display = "none";

    try {
        let response;
        try {
            response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code }),
            });
        } catch (netErr) {
            showError("Could not connect to the server. Make sure the Flask app is running.");
            return;
        }

        let data;
        try {
            data = await response.json();
        } catch (jsonErr) {
            showError(`Server error (${response.status}): The server returned an invalid response.`);
            return;
        }

        if (!response.ok) {
            showError(data.error || `Server error (${response.status}): Something went wrong.`);
            return;
        }

        // Display results
        showResult(data);
    } catch (err) {
        showError(err.message || "An unexpected error occurred.");
    } finally {
        setLoading(false);
    }
}

// ============================================================
// Source labels and icons
// ============================================================
const SOURCE_LABELS = {
    "static_analysis": "Python Parser",
    "bracket_check": "Bracket Check",
    "pyflakes": "Pyflakes",
    "ast_rules": "AST Rule",
    "runtime_analysis": "Runtime",
    "codebert": "CodeBERT ML",
};

const SOURCE_ICONS = {
    "static_analysis": "🔍",
    "bracket_check": "🔗",
    "pyflakes": "🐍",
    "ast_rules": "📐",
    "runtime_analysis": "▶️",
    "codebert": "🤖",
};

// Bug type severity colors
function getSeverityClass(type) {
    const errorTypes = [
        "syntax_error", "indentation_error", "mismatched_bracket",
        "unclosed_bracket", "unterminated_string",
    ];
    const warningTypes = [
        "off_by_one", "wrong_operator", "wrong_comparison",
        "wrong_loop_condition", "wrong_boolean_logic",
        "return_inside_loop_prematurely", "mutable_default_argument",
    ];
    const infoTypes = [
        "undefined_variable", "missing_return", "null_reference",
        "unbound_local_error",
    ];

    if (errorTypes.includes(type)) return "severity-error";
    if (warningTypes.includes(type)) return "severity-warning";
    if (infoTypes.includes(type)) return "severity-info";
    if (type === "no_bug") return "severity-clean";
    return "severity-warning";
}

// ============================================================
// Display result (multi-issue)
// ============================================================
function showResult(data) {
    const resultTitle = document.getElementById("resultTitle");
    const issueCountBadge = document.getElementById("issueCountBadge");
    const summaryText = document.getElementById("summaryText");
    const issuesList = document.getElementById("issuesList");
    const notesSection = document.getElementById("notesSection");
    const notesList = document.getElementById("notesList");
    const mlSection = document.getElementById("mlSection");
    const top3List = document.getElementById("top3List");

    const issues = data.issues || [];
    const notes = data.notes || [];
    const count = data.issue_count || 0;

    // --- Title & badge ---
    if (count === 0) {
        resultTitle.textContent = "✅ No Issues Found";
        issueCountBadge.textContent = "";
        issueCountBadge.className = "issue-count-badge badge-clean";
    } else {
        resultTitle.textContent = "Analysis Result";
        issueCountBadge.textContent = `${count} issue${count !== 1 ? "s" : ""} found`;
        issueCountBadge.className = "issue-count-badge badge-issues";
    }

    // --- Summary ---
    summaryText.textContent = data.summary || "";

    // --- Issue cards ---
    issuesList.innerHTML = "";
    issues.forEach((issue, idx) => {
        const card = document.createElement("div");
        const sevClass = getSeverityClass(issue.type);
        card.className = `issue-card ${sevClass}`;

        const sourceLabel = SOURCE_LABELS[issue.source] || issue.source;
        const sourceIcon = SOURCE_ICONS[issue.source] || "🔎";
        const bugLabel = issue.type.replace(/_/g, " ");
        const lineText = issue.line === "N/A" ? "" : `Line ${issue.line}`;

        card.innerHTML = `
            <div class="issue-card-header">
                <span class="issue-number">#${idx + 1}</span>
                <span class="issue-type-badge ${sevClass}">${bugLabel}</span>
                ${lineText ? `<span class="issue-line">${lineText}</span>` : ""}
                <span class="issue-source">${sourceIcon} ${sourceLabel}</span>
            </div>
            <p class="issue-message">${escapeHtml(issue.message)}</p>
        `;
        issuesList.appendChild(card);
    });

    // --- Notes (warnings) ---
    if (notes.length > 0) {
        notesSection.style.display = "block";
        notesList.innerHTML = "";
        notes.forEach(note => {
            const li = document.createElement("li");
            li.textContent = `Line ${note.line}: ${note.message}`;
            notesList.appendChild(li);
        });
    } else {
        notesSection.style.display = "none";
    }

    // --- ML top-3 ---
    if (data.ml_top3 && data.ml_top3.length > 0) {
        mlSection.style.display = "block";
        top3List.innerHTML = "";
        data.ml_top3.forEach(pred => {
            const li = document.createElement("li");
            li.textContent = `${pred.label.replace(/_/g, " ")} (${pred.confidence}%)`;
            top3List.appendChild(li);
        });
    } else {
        mlSection.style.display = "none";
    }

    // Show section with animation
    resultSection.style.display = "block";
    errorSection.style.display = "none";

    // Scroll to result
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ============================================================
// Utility: escape HTML
// ============================================================
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// Display error
// ============================================================
function showError(message) {
    document.getElementById("errorText").textContent = message;
    errorSection.style.display = "block";
    resultSection.style.display = "none";
}

// ============================================================
// Loading state
// ============================================================
function setLoading(loading) {
    analyzeBtn.disabled = loading;
    analyzeBtn.querySelector(".btn-text").style.display = loading ? "none" : "";
    analyzeBtn.querySelector(".btn-loading").style.display = loading ? "" : "none";
}

// ============================================================
// Clear all
// ============================================================
function clearAll() {
    codeInput.value = "";
    codeInput.dispatchEvent(new Event("input"));
    resultSection.style.display = "none";
    errorSection.style.display = "none";
    codeInput.focus();
}
