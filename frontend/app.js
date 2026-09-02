let lastAssessment = null;

const API_URL = "http://127.0.0.1:8000/assess";
const TRANSACTIONS_URL = "http://127.0.0.1:8000/transactions";


// =====================================================
// DOM ELEMENTS
// =====================================================

const riskAssessmentNav = document.getElementById("riskAssessmentNav");
const transactionsNav = document.getElementById("transactionsNav");
const analyticsNav = document.getElementById("analyticsNav");

const assessmentPage = document.getElementById("assessmentPage");
const transactionsPage = document.getElementById("transactionsPage");

const results = document.getElementById("results");
const emptyState = document.getElementById("emptyState");


// =====================================================
// NAVIGATION
// =====================================================

riskAssessmentNav.addEventListener("click", showRiskAssessment);
transactionsNav.addEventListener("click", showTransactions);
analyticsNav.addEventListener("click", showAnalytics);


// =====================================================
// SHOW RISK ASSESSMENT
// =====================================================

function showRiskAssessment() {

    riskAssessmentNav.classList.add("active");
    transactionsNav.classList.remove("active");
    analyticsNav.classList.remove("active");

    assessmentPage.classList.remove("hidden");
    transactionsPage.classList.add("hidden");

    // Restore only the assessment from the current browser session.
    // Do NOT fetch the old transaction history here.
    if (lastAssessment) {
        displayResults(lastAssessment, false);
    } else {
        results.classList.add("hidden");
        emptyState.classList.remove("hidden");
    }
}


// =====================================================
// SHOW TRANSACTIONS
// =====================================================

async function showTransactions() {

    riskAssessmentNav.classList.remove("active");
    transactionsNav.classList.add("active");
    analyticsNav.classList.remove("active");

    assessmentPage.classList.add("hidden");
    transactionsPage.classList.remove("hidden");

    await loadTransactions();
}


// =====================================================
// ANALYTICS
// =====================================================

function showAnalytics() {

    alert("Analytics dashboard coming soon.");
}


// =====================================================
// ASSESS TRANSACTION
// =====================================================

async function assessTransaction() {

    const button = document.getElementById("assessButton");

    const customerId =
        document.getElementById("customerId").value.trim();

    const terminalId =
        document.getElementById("terminalId").value.trim();

    const amount =
        document.getElementById("amount").value.trim();

    const timestamp =
        document.getElementById("timestamp").value;


    // -------------------------------------------------
    // VALIDATION
    // -------------------------------------------------

    if (!customerId || !terminalId || !amount || !timestamp) {

        alert("Please fill in all transaction details.");
        return;
    }

    if (Number(customerId) <= 0 || Number(terminalId) <= 0) {

        alert("Customer ID and Terminal ID must be valid numbers.");
        return;
    }

    if (Number(amount) <= 0) {

        alert("Transaction amount must be greater than zero.");
        return;
    }


    // -------------------------------------------------
    // BUTTON STATE
    // -------------------------------------------------

    button.disabled = true;

    const buttonText =
        button.querySelector("span:first-child");

    if (buttonText) {
        buttonText.textContent = "Analysing...";
    }


    // -------------------------------------------------
    // API REQUEST
    // -------------------------------------------------

    try {

        const response = await fetch(API_URL, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                customer_id: Number(customerId),

                terminal_id: Number(terminalId),

                amount: Number(amount),

                timestamp: timestamp.replace("T", " ")

            })
        });


        // -------------------------------------------------
        // API ERROR
        // -------------------------------------------------

        if (!response.ok) {

            let errorMessage =
                `API returned ${response.status}`;

            try {

                const errorData =
                    await response.json();

                if (errorData.detail) {

                    errorMessage +=
                        `: ${errorData.detail}`;
                }

            } catch {
                // Ignore JSON parsing failure
            }

            throw new Error(errorMessage);
        }


        // -------------------------------------------------
        // GET RESULT
        // -------------------------------------------------

        const data = await response.json();


        // IMPORTANT:
        // Store the current assessment in memory.
        // This allows it to remain visible when navigating
        // between Risk Assessment and Transactions.
        lastAssessment = data;

        displayResults(data, true);

    }


    // -------------------------------------------------
    // ERROR HANDLING
    // -------------------------------------------------

    catch (error) {

        console.error(
            "RiskLens API error:",
            error
        );

        alert(
            "Unable to connect to the RiskLens API.\n\n" +
            error.message
        );
    }


    // -------------------------------------------------
    // RESET BUTTON
    // -------------------------------------------------

    finally {

        button.disabled = false;

        if (buttonText) {
            buttonText.textContent = "Assess Transaction";
        }
    }
}


// =====================================================
// DISPLAY RISK RESULTS
// =====================================================

function displayResults(data, shouldScroll = false) {

    if (!data) {
        return;
    }


    // -------------------------------------------------
    // SAVE CURRENT ASSESSMENT
    // -------------------------------------------------

    lastAssessment = data;


    // -------------------------------------------------
    // SHOW RISK ASSESSMENT PAGE
    // -------------------------------------------------

    assessmentPage.classList.remove("hidden");
    transactionsPage.classList.add("hidden");

    riskAssessmentNav.classList.add("active");
    transactionsNav.classList.remove("active");
    analyticsNav.classList.remove("active");


    // -------------------------------------------------
    // SHOW RESULTS
    // -------------------------------------------------

    results.classList.remove("hidden");
    emptyState.classList.add("hidden");


    // =================================================
    // RISK DATA
    // =================================================

    const riskScore = Number(
        data.risk_percentage ??
        ((data.risk_score ?? 0) * 100)
    );

    const riskLevel =
        data.risk_level || "UNKNOWN";

    const action =
        data.action || "REVIEW";


    document.getElementById("riskScore").textContent =
        `${riskScore.toFixed(2)}%`;

    document.getElementById("riskLevel").textContent =
        riskLevel;

    document.getElementById("riskBadge").textContent =
        riskLevel;

    document.getElementById("action").textContent =
        action;


    // =================================================
    // RISK BAR
    // =================================================

    const riskBar =
        document.getElementById("riskBar");

    const safeRiskScore =
        Math.min(
            Math.max(riskScore, 0),
            100
        );

    riskBar.style.width =
        `${safeRiskScore}%`;


    // =================================================
    // RISK COLOUR
    // =================================================

    let riskColor;

    if (riskLevel === "HIGH") {

        riskColor = "#dc3545";

    } else if (riskLevel === "MEDIUM") {

        riskColor = "#e08a00";

    } else {

        riskColor = "#159957";
    }


    document.getElementById("riskScore").style.color =
        riskColor;

    document.getElementById("riskLevel").style.color =
        riskColor;

    document.getElementById("riskBadge").style.background =
        `${riskColor}18`;

    document.getElementById("riskBadge").style.color =
        riskColor;

    riskBar.style.background =
        riskColor;


    // =================================================
    // EXPLANATION
    // =================================================

    const explanation =
        data.explanation || {};


    document.getElementById("summary").textContent =
        explanation.summary ||
        "Transaction assessed using behavioural risk signals.";


    // =================================================
    // REASONS
    // =================================================

    const reasonsContainer =
        document.getElementById("reasons");

    reasonsContainer.innerHTML = "";

    const reasons =
        Array.isArray(explanation.reasons)
            ? explanation.reasons
            : [];


    if (reasons.length === 0) {

        const reason =
            document.createElement("div");

        reason.className = "reason";

        reason.textContent =
            "No major behavioural anomalies detected.";

        reasonsContainer.appendChild(reason);

    } else {

        reasons.forEach(text => {

            const reason =
                document.createElement("div");

            reason.className = "reason";

            reason.textContent = text;

            reasonsContainer.appendChild(reason);
        });
    }


    // =================================================
    // BEHAVIOURAL SIGNALS
    // =================================================

    const signalsContainer =
        document.getElementById("signals");

    signalsContainer.innerHTML = "";


    const signals =
        data.behavioural_signals || {};


    const signalLabels = {

        customer_transaction_count:
            "Customer transactions",

        customer_average_amount:
            "Customer avg. amount",

        customer_amount_ratio:
            "Customer amount ratio",

        customer_transactions_1h:
            "Customer transactions (1h)",

        customer_transactions_24h:
            "Customer transactions (24h)",

        terminal_transaction_count:
            "Terminal transactions",

        terminal_average_amount:
            "Terminal avg. amount",

        terminal_amount_ratio:
            "Terminal amount ratio",

        customer_amount_zscore:
            "Customer amount Z-score",

        known_customer_terminal:
            "Known customer terminal"
    };


    Object.entries(signals).forEach(
        ([key, value]) => {

            if (!(key in signalLabels)) {
                return;
            }


            const card =
                document.createElement("div");

            card.className = "signal";


            const label =
                document.createElement("span");

            label.className =
                "signal-label";

            label.textContent =
                signalLabels[key];


            const valueElement =
                document.createElement("span");

            valueElement.className =
                "signal-value";


            if (typeof value === "boolean") {

                valueElement.textContent =
                    value ? "Yes" : "No";

            } else if (typeof value === "number") {

                valueElement.textContent =
                    Number.isInteger(value)
                        ? String(value)
                        : value.toFixed(2);

            } else {

                valueElement.textContent =
                    String(value);
            }


            card.appendChild(label);
            card.appendChild(valueElement);

            signalsContainer.appendChild(card);
        }
    );


    // =================================================
    // SCROLL TO RESULTS
    // =================================================

    if (shouldScroll) {

        results.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }
}


// =====================================================
// LOAD TRANSACTION HISTORY
// =====================================================

async function loadTransactions() {

    const tableBody =
        document.getElementById("transactionTableBody");

    const countText =
        document.getElementById("transactionCount");


    if (!tableBody) {

        console.error(
            "Transaction table not found."
        );

        return;
    }


    // -------------------------------------------------
    // LOADING STATE
    // -------------------------------------------------

    tableBody.innerHTML = `
        <tr>
            <td colspan="6">
                Loading transactions...
            </td>
        </tr>
    `;


    try {

        const response =
            await fetch(TRANSACTIONS_URL);


        if (!response.ok) {

            throw new Error(
                `API returned ${response.status}`
            );
        }


        const data =
            await response.json();


        const transactions =
            Array.isArray(data.transactions)
                ? data.transactions
                : [];


        // -------------------------------------------------
        // COUNT
        // -------------------------------------------------

        if (countText) {

            countText.textContent =
                `${transactions.length} transaction` +
                `${transactions.length === 1 ? "" : "s"} assessed`;
        }


        tableBody.innerHTML = "";


        // -------------------------------------------------
        // EMPTY HISTORY
        // -------------------------------------------------

        if (transactions.length === 0) {

            tableBody.innerHTML = `
                <tr>
                    <td colspan="6">
                        No transactions assessed yet.
                    </td>
                </tr>
            `;

            return;
        }


        // -------------------------------------------------
        // DISPLAY TRANSACTIONS
        // -------------------------------------------------

        transactions
            .slice()
            .reverse()
            .forEach(item => {

                const row =
                    document.createElement("tr");


                const transaction =
                    item.transaction || {};


                const customer =
                    transaction.customer_id ?? "—";


                const terminal =
                    transaction.terminal_id ?? "—";


                const amount =
                    Number(
                        transaction.amount ?? 0
                    );


                const risk =
                    Number(
                        item.risk_percentage ??
                        ((item.risk_score ?? 0) * 100)
                    );


                const level =
                    item.risk_level ?? "—";


                const action =
                    item.action ?? "—";


                row.innerHTML = `
                    <td>${customer}</td>
                    <td>${terminal}</td>
                    <td>₹${amount.toFixed(2)}</td>
                    <td>${risk.toFixed(2)}%</td>
                    <td>${level}</td>
                    <td>${action}</td>
                `;


                tableBody.appendChild(row);
            });

    }


    // -------------------------------------------------
    // ERROR
    // -------------------------------------------------

    catch (error) {

        console.error(
            "Transaction history error:",
            error
        );


        tableBody.innerHTML = `
            <tr>
                <td colspan="6">
                    Unable to load transaction history.
                </td>
            </tr>
        `;
    }
}


// =====================================================
// INITIAL PAGE STATE
// =====================================================

// The page must ALWAYS begin with a blank assessment.
// Previous transactions are displayed only on the
// Transactions page, never automatically restored here.

function initializePage() {

    lastAssessment = null;

    results.classList.add("hidden");
    emptyState.classList.remove("hidden");

    assessmentPage.classList.remove("hidden");
    transactionsPage.classList.add("hidden");

    riskAssessmentNav.classList.add("active");
    transactionsNav.classList.remove("active");
    analyticsNav.classList.remove("active");
}


// =====================================================
// START APPLICATION
// =====================================================

initializePage();