// pii_scanner.cpp - Data element type scanner (PII / customer / financial data discovery)
//
// Build:  g++ -std=c++17 -O2 -Wall -Wextra -o pii_scanner pii_scanner.cpp
// Usage:  ./pii_scanner [options] <file|dir|-> ...
//
// Detects: SSN, credit cards (Luhn + brand), email, US/intl phone, IBAN (mod-97),
//          ABA routing numbers (checksum), bank accounts (context), street address,
//          PO box, ZIP, DOB, EIN, passport, driver license, customer names, IPv4,
//          AWS keys, private keys.
//
// Matches are masked in output by default. Use --show-raw only in a safe environment.

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using std::string;
using std::vector;

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
static inline bool isAlnum(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; }
static inline bool isDigit(char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }

static string toLower(string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static string digitsOnly(const string& s) {
    string d;
    for (char c : s) if (isDigit(c)) d += c;
    return d;
}

static bool allSame(const string& d) {
    return !d.empty() && std::all_of(d.begin(), d.end(), [&](char c) { return c == d[0]; });
}

static bool hasSeparator(const string& s) {
    return s.find_first_of("()+ .-") != string::npos;
}

// ----------------------------------------------------------------------------
// Validators
// ----------------------------------------------------------------------------
static bool luhnOk(const string& d) {
    int sum = 0;
    bool alt = false;
    for (int i = static_cast<int>(d.size()) - 1; i >= 0; --i) {
        int n = d[i] - '0';
        if (alt) { n *= 2; if (n > 9) n -= 9; }
        sum += n;
        alt = !alt;
    }
    return sum % 10 == 0;
}

static string cardBrand(const string& d) {
    size_t n = d.size();
    auto pre = [&](size_t len) { return std::stoi(d.substr(0, len)); };
    if (d[0] == '4' && (n == 13 || n == 16 || n == 19)) return "Visa";
    if (n == 16) {
        int p2 = pre(2), p4 = pre(4);
        if ((p2 >= 51 && p2 <= 55) || (p4 >= 2221 && p4 <= 2720)) return "Mastercard";
    }
    if (n == 15 && (pre(2) == 34 || pre(2) == 37)) return "Amex";
    if (n >= 16 && n <= 19) {
        if (pre(4) == 6011 || pre(2) == 65 || (pre(3) >= 644 && pre(3) <= 649)) return "Discover";
        if (pre(2) == 62) return "UnionPay";
        if (pre(4) >= 3528 && pre(4) <= 3589) return "JCB";
    }
    if (n == 14 && ((pre(3) >= 300 && pre(3) <= 305) || pre(2) == 36 || pre(2) == 38)) return "Diners";
    return "";
}

static bool validCard(const string& m, bool) {
    string d = digitsOnly(m);
    if (d.size() < 13 || d.size() > 19 || allSame(d) || !luhnOk(d)) return false;
    return !cardBrand(d).empty();
}
static string cardSubtype(const string& m) { return cardBrand(digitsOnly(m)); }

static bool validSSN(const string& m, bool) {
    string d = digitsOnly(m);
    if (d.size() != 9 || allSame(d)) return false;
    int area = std::stoi(d.substr(0, 3));
    int group = std::stoi(d.substr(3, 2));
    int serial = std::stoi(d.substr(5, 4));
    if (area == 0 || area == 666 || area >= 900) return false;
    if (group == 0 || serial == 0) return false;
    if (d == "078051120" || d == "219099999") return false;  // famous advertising SSNs
    return true;
}

static bool validABA(const string& m, bool) {
    string d = digitsOnly(m);
    if (d.size() != 9 || allSame(d)) return false;
    auto v = [&](int i) { return d[i] - '0'; };
    int sum = 3 * (v(0) + v(3) + v(6)) + 7 * (v(1) + v(4) + v(7)) + (v(2) + v(5) + v(8));
    return sum % 10 == 0;
}

static bool validIBAN(const string& m, bool) {
    string s;
    for (char c : m) if (c != ' ') s += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (s.size() < 15 || s.size() > 34) return false;
    string r = s.substr(4) + s.substr(0, 4);
    int rem = 0;
    for (char c : r) {
        if (isDigit(c)) rem = (rem * 10 + (c - '0')) % 97;
        else if (c >= 'A' && c <= 'Z') rem = (rem * 100 + (c - 'A' + 10)) % 97;
        else return false;
    }
    return rem == 1;
}

static bool validUSPhone(const string& m, bool ctx) {
    if (!hasSeparator(m) && !ctx) return false;  // bare 10-digit numbers need context
    string d = digitsOnly(m);
    if (d.size() == 11 && d[0] == '1') d = d.substr(1);
    if (d.size() != 10 || allSame(d)) return false;
    return d[0] >= '2' && d[3] >= '2';           // NANP: area code & exchange start 2-9
}

static bool validIntlPhone(const string& m, bool) {
    size_t n = digitsOnly(m).size();
    return n >= 8 && n <= 15;
}

static bool validIPv4(const string& m, bool) {
    std::stringstream ss(m);
    string part;
    int count = 0;
    while (std::getline(ss, part, '.')) {
        if (part.empty() || part.size() > 3) return false;
        if (part.size() > 1 && part[0] == '0') return false;
        if (std::stoi(part) > 255) return false;
        ++count;
    }
    return count == 4 && m != "0.0.0.0" && m != "127.0.0.1";
}

static bool hasDigit(const string& m, bool) {
    return std::any_of(m.begin(), m.end(), [](char c) { return isDigit(c); });
}

static bool plausibleAccount(const string& m, bool) {
    return !allSame(m) && m.size() >= 6 && m.size() <= 17;
}

// ----------------------------------------------------------------------------
// Rule definitions
// ----------------------------------------------------------------------------
enum class Mask { Tail4, Email, Words, Full };

struct Rule {
    string id, category, description;
    std::regex re;
    int group = 0;
    bool bounded = true;                 // require non-alnum chars around the match
    bool hasCtx = false, needCtx = false;
    std::regex ctx;                      // keyword context searched on the lowercased line
    double conf = 0.7, ctxBoost = 0.15;
    std::function<bool(const string&, bool)> validate;
    std::function<string(const string&)> subtype;
    Mask mask = Mask::Tail4;
};

static Rule mk(string id, string cat, string desc, const char* pat, double conf, bool icase = false) {
    Rule r;
    r.id = std::move(id);
    r.category = std::move(cat);
    r.description = std::move(desc);
    auto flags = std::regex::ECMAScript | std::regex::optimize;
    if (icase) flags |= std::regex::icase;
    r.re = std::regex(pat, flags);
    r.conf = conf;
    return r;
}

static void setCtx(Rule& r, const char* pat, bool required) {
    r.ctx = std::regex(pat, std::regex::ECMAScript | std::regex::icase);
    r.hasCtx = true;
    r.needCtx = required;
}

static vector<Rule> buildRules() {
    vector<Rule> v;

    { auto& r = v.emplace_back(mk("ssn", "Government ID", "US Social Security Number (formatted)",
                                  R"(\d{3}-\d{2}-\d{4})", 0.80));
      r.validate = validSSN; setCtx(r, R"(ssn|social security|soc\.? ?sec|tax ?id)", false); }

    { auto& r = v.emplace_back(mk("ssn_plain", "Government ID", "US SSN (9 digits, needs context)",
                                  R"(\d{9})", 0.70));
      r.validate = validSSN; setCtx(r, R"(ssn|social security|soc\.? ?sec)", true); }

    { auto& r = v.emplace_back(mk("ein", "Government ID", "US Employer Identification Number",
                                  R"(\d{2}-\d{7})", 0.65));
      setCtx(r, R"(\bein\b|\bfein\b|\btin\b|tax id|employer id)", true); }

    { auto& r = v.emplace_back(mk("passport", "Government ID", "Passport number (needs context)",
                                  R"([A-Z0-9]{6,9})", 0.60));
      r.validate = hasDigit; setCtx(r, R"(passport)", true); }

    { auto& r = v.emplace_back(mk("driver_license", "Government ID", "Driver license number (needs context)",
                                  R"([A-Z0-9]{5,15})", 0.55));
      r.validate = hasDigit; setCtx(r, R"(driver'?s? ?licen[sc]e|\bdl ?(#|no|num))", true); }

    { auto& r = v.emplace_back(mk("credit_card", "Financial", "Payment card number (Luhn + brand)",
                                  R"(\d(?:[ -]?\d){12,18})", 0.90));
      r.validate = validCard; r.subtype = cardSubtype; setCtx(r, R"(card|visa|mastercard|amex|cc ?(num|#)|pan\b)", false); }

    { auto& r = v.emplace_back(mk("iban", "Financial", "IBAN (mod-97 checked)",
                                  R"([A-Z]{2}\d{2}(?: ?[A-Za-z0-9]{4}){2,7}(?: ?[A-Za-z0-9]{1,3})?)", 0.92));
      r.validate = validIBAN; }

    { auto& r = v.emplace_back(mk("aba_routing", "Financial", "ABA routing number (checksum, needs context)",
                                  R"(\d{9})", 0.75));
      r.validate = validABA; setCtx(r, R"(routing|\baba\b|\brtn\b|transit)", true); }

    { auto& r = v.emplace_back(mk("bank_account", "Financial", "Bank account number (needs context)",
                                  R"(\d{6,17})", 0.60));
      r.validate = plausibleAccount; setCtx(r, R"(account|acct|a/c|checking|savings|bank)", true); }

    { auto& r = v.emplace_back(mk("email", "Contact", "Email address",
                                  R"([A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})", 0.90));
      r.bounded = false; r.mask = Mask::Email; }

    { auto& r = v.emplace_back(mk("phone_us", "Contact", "US/Canada phone number",
                                  R"((?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4})", 0.72));
      r.validate = validUSPhone; setCtx(r, R"(phone|tel\b|mobile|cell|fax|call|contact)", false); }

    { auto& r = v.emplace_back(mk("phone_intl", "Contact", "International phone number (E.164-like)",
                                  R"(\+\d{1,3}(?:[ .-]?\d){6,12})", 0.70));
      r.validate = validIntlPhone; setCtx(r, R"(phone|tel\b|mobile|cell|fax|call|contact)", false); }

    { auto& r = v.emplace_back(mk("street_address", "Address", "US street address",
        R"(\d{1,6}\s+(?:[A-Za-z0-9.'-]+\s+){1,4}(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|court|ct|way|place|pl|terrace|ter|parkway|pkwy|circle|cir)\b\.?)",
        0.70, true));
      r.mask = Mask::Words; setCtx(r, R"(address|addr|street|ship|bill|resid)", false); }

    { auto& r = v.emplace_back(mk("po_box", "Address", "PO Box", R"(P\.?\s?O\.?\s?Box\s+\d+)", 0.70, true));
      r.mask = Mask::Words; }

    { auto& r = v.emplace_back(mk("zip", "Address", "US ZIP code (needs context)",
                                  R"(\d{5}(?:-\d{4})?)", 0.50));
      setCtx(r, R"(zip|postal|post code|city|state|address)", true); }

    { auto& r = v.emplace_back(mk("dob", "Personal", "Date of birth (needs context)",
        R"((?:(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}|(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])))",
        0.70));
      setCtx(r, R"(dob|birth|born|d\.o\.b)", true); }

    { auto& r = v.emplace_back(mk("person_name", "Personal", "Labeled customer / person name",
        R"((?:[Cc]ustomer|[Cc]lient|[Cc]ardholder|[Ff]ull|[Aa]ccount ?[Hh]older|[Pp]atient)[ _-]?[Nn]ame\s*[:=]\s*([A-Z][a-z]+(?: [A-Z][a-z'-]+){1,2}))",
        0.75));
      r.group = 1; r.bounded = false; r.mask = Mask::Words; }

    { auto& r = v.emplace_back(mk("ipv4", "Network", "IPv4 address",
                                  R"((?:\d{1,3}\.){3}\d{1,3})", 0.55));
      r.validate = validIPv4; }

    { auto& r = v.emplace_back(mk("aws_key", "Secret", "AWS access key ID",
                                  R"((?:AKIA|ASIA)[A-Z0-9]{16})", 0.95)); (void)r; }

    { auto& r = v.emplace_back(mk("private_key", "Secret", "Private key header",
                                  R"(-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----)", 0.99));
      r.bounded = false; r.mask = Mask::Full; }

    return v;
}

// ----------------------------------------------------------------------------
// Findings
// ----------------------------------------------------------------------------
struct Candidate {
    const Rule* rule;
    size_t pos, len;
    double conf;
    string text;
};

struct Finding {
    string file;
    size_t line, col;
    string id, category, subtype;
    double conf;
    string masked, raw;
};

struct Options {
    bool json = false, csv = false, raw = false, listRules = false, help = false;
    double minConf = 0.5;
    std::set<string> only;
    std::uintmax_t maxBytes = 50ull * 1024 * 1024;
    vector<string> paths;
};

static string maskValue(const Rule& r, const string& s) {
    switch (r.mask) {
        case Mask::Email: {
            auto at = s.find('@');
            return (at == string::npos || at == 0) ? "***" : s.substr(0, 1) + "***" + s.substr(at);
        }
        case Mask::Words: {
            string o;
            bool start = true;
            for (char c : s) {
                if (isAlnum(c)) { o += start ? c : '*'; start = false; }
                else { o += c; start = true; }
            }
            return o;
        }
        case Mask::Full: return "[REDACTED]";
        case Mask::Tail4:
        default: {
            size_t n = 0;
            for (char c : s) if (isAlnum(c)) ++n;
            string o;
            size_t idx = 0;
            for (char c : s) {
                if (isAlnum(c)) { o += (idx + 4 >= n) ? c : '*'; ++idx; }
                else o += c;
            }
            return o;
        }
    }
}

static void scanLine(const string& line, const vector<Rule>& rules, const Options& opt,
                     vector<Candidate>& out) {
    string low;
    bool lowDone = false;
    for (const auto& rule : rules) {
        if (!opt.only.empty() && !opt.only.count(rule.id) && !opt.only.count(toLower(rule.category))) continue;

        bool ctxChecked = false, ctxHit = false;
        auto getCtx = [&]() {
            if (!rule.hasCtx) return false;
            if (!ctxChecked) {
                if (!lowDone) { low = toLower(line); lowDone = true; }
                ctxHit = std::regex_search(low, rule.ctx);
                ctxChecked = true;
            }
            return ctxHit;
        };
        if (rule.needCtx && !getCtx()) continue;

        for (std::sregex_iterator it(line.begin(), line.end(), rule.re), end; it != end; ++it) {
            const auto& g = (*it)[rule.group];
            if (!g.matched) continue;
            size_t pos = static_cast<size_t>(g.first - line.begin());
            size_t len = static_cast<size_t>(g.length());
            if (rule.bounded) {
                if (pos > 0 && isAlnum(line[pos - 1])) continue;
                if (pos + len < line.size() && isAlnum(line[pos + len])) continue;
            }
            string text = g.str();
            bool ctx = getCtx();
            if (rule.validate && !rule.validate(text, ctx)) continue;
            double c = rule.conf + ((ctx && rule.hasCtx) ? rule.ctxBoost : 0.0);
            c = std::min(c, 0.99);
            if (c < opt.minConf) continue;
            out.push_back({&rule, pos, len, c, std::move(text)});
        }
    }
}

// Keep the highest-confidence candidate whenever matches overlap.
static vector<Candidate> resolveOverlaps(vector<Candidate> c) {
    std::sort(c.begin(), c.end(), [](const Candidate& a, const Candidate& b) {
        if (a.conf != b.conf) return a.conf > b.conf;
        if (a.len != b.len) return a.len > b.len;
        return a.pos < b.pos;
    });
    vector<Candidate> kept;
    for (auto& x : c) {
        bool overlap = std::any_of(kept.begin(), kept.end(), [&](const Candidate& k) {
            return x.pos < k.pos + k.len && k.pos < x.pos + x.len;
        });
        if (!overlap) kept.push_back(std::move(x));
    }
    std::sort(kept.begin(), kept.end(), [](const Candidate& a, const Candidate& b) { return a.pos < b.pos; });
    return kept;
}

static void scanStream(std::istream& in, const string& name, const vector<Rule>& rules,
                       const Options& opt, vector<Finding>& findings) {
    string line;
    size_t lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() > 1'000'000) continue;  // skip pathological (minified) lines
        vector<Candidate> cands;
        scanLine(line, rules, opt, cands);
        if (cands.empty()) continue;
        for (const auto& c : resolveOverlaps(std::move(cands))) {
            Finding f;
            f.file = name;
            f.line = lineNo;
            f.col = c.pos + 1;
            f.id = c.rule->id;
            f.category = c.rule->category;
            f.subtype = c.rule->subtype ? c.rule->subtype(c.text) : "";
            f.conf = c.conf;
            f.masked = maskValue(*c.rule, c.text);
            if (opt.raw) f.raw = c.text;
            findings.push_back(std::move(f));
        }
    }
}

static bool looksBinary(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    char buf[4096];
    f.read(buf, sizeof(buf));
    std::streamsize n = f.gcount();
    for (std::streamsize i = 0; i < n; ++i) if (buf[i] == '\0') return true;
    return false;
}

// ----------------------------------------------------------------------------
// Output
// ----------------------------------------------------------------------------
static string jsonEscape(const string& s) {
    string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += static_cast<char>(c);
        }
    }
    return o;
}

static string csvEscape(const string& s) {
    string o = "\"";
    for (char c : s) { if (c == '"') o += '"'; o += c; }
    return o + "\"";
}

static string typeLabel(const Finding& f) {
    return f.subtype.empty() ? f.id : f.id + "/" + f.subtype;
}

static void printFindings(const vector<Finding>& fs_, const Options& opt) {
    if (opt.json) {
        std::cout << "[\n";
        for (size_t i = 0; i < fs_.size(); ++i) {
            const auto& f = fs_[i];
            std::cout << "  {\"file\":\"" << jsonEscape(f.file) << "\",\"line\":" << f.line
                      << ",\"column\":" << f.col << ",\"type\":\"" << f.id << "\",\"category\":\""
                      << f.category << "\",\"subtype\":\"" << f.subtype << "\",\"confidence\":"
                      << std::fixed << std::setprecision(2) << f.conf << ",\"value\":\""
                      << jsonEscape(opt.raw ? f.raw : f.masked) << "\"}" << (i + 1 < fs_.size() ? "," : "") << "\n";
        }
        std::cout << "]\n";
    } else if (opt.csv) {
        std::cout << "file,line,column,type,category,subtype,confidence,value\n";
        for (const auto& f : fs_)
            std::cout << csvEscape(f.file) << ',' << f.line << ',' << f.col << ',' << f.id << ','
                      << csvEscape(f.category) << ',' << f.subtype << ',' << std::fixed
                      << std::setprecision(2) << f.conf << ',' << csvEscape(opt.raw ? f.raw : f.masked) << "\n";
    } else {
        for (const auto& f : fs_)
            std::cout << f.file << ':' << f.line << ':' << f.col << "  [" << std::left << std::setw(16)
                      << typeLabel(f) << "] conf=" << std::fixed << std::setprecision(2) << f.conf << "  "
                      << (opt.raw ? f.raw : f.masked) << "\n";
    }
}

static void printSummary(const vector<Finding>& findings, size_t files, size_t skipped, std::ostream& os) {
    std::map<string, size_t> byType, byCat;
    std::map<string, std::set<string>> filesByType;
    for (const auto& f : findings) {
        ++byType[f.id];
        ++byCat[f.category];
        filesByType[f.id].insert(f.file);
    }
    os << "\n==== Summary ====\n";
    os << "Files scanned: " << files << "  (skipped: " << skipped << ")\n";
    os << "Total findings: " << findings.size() << "\n";
    if (findings.empty()) return;
    os << "\nBy data element type:\n";
    for (const auto& [id, n] : byType)
        os << "  " << std::left << std::setw(18) << id << std::right << std::setw(7) << n
           << "  in " << filesByType[id].size() << " file(s)\n";
    os << "\nBy category:\n";
    for (const auto& [cat, n] : byCat)
        os << "  " << std::left << std::setw(18) << cat << std::right << std::setw(7) << n << "\n";
}

static void usage(const char* exe) {
    std::cout <<
        "Data element type scanner (PII / customer / financial data discovery)\n\n"
        "Usage: " << exe << " [options] <file|dir|-> ...\n\n"
        "Options:\n"
        "  --json               JSON output\n"
        "  --csv                CSV output\n"
        "  --show-raw           Show unmasked values (handle output carefully!)\n"
        "  --min-conf <0-1>     Minimum confidence to report (default 0.50)\n"
        "  --types a,b,c        Only these rule ids or categories (e.g. ssn,financial)\n"
        "  --max-size <MB>      Skip files larger than this (default 50)\n"
        "  --list-rules         List detection rules\n"
        "  -h, --help           This help\n\n"
        "Use '-' to read from stdin. Directories are scanned recursively.\n"
        "Exit code: 0 = nothing found, 1 = findings, 2 = error.\n";
}

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        auto need = [&](const char* flag) -> string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << flag << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--json") opt.json = true;
        else if (a == "--csv") opt.csv = true;
        else if (a == "--show-raw") opt.raw = true;
        else if (a == "--list-rules") opt.listRules = true;
        else if (a == "-h" || a == "--help") opt.help = true;
        else if (a == "--min-conf") opt.minConf = std::stod(need("--min-conf"));
        else if (a == "--max-size") opt.maxBytes = static_cast<std::uintmax_t>(std::stod(need("--max-size")) * 1024 * 1024);
        else if (a == "--types") {
            std::stringstream ss(need("--types"));
            string t;
            while (std::getline(ss, t, ',')) if (!t.empty()) opt.only.insert(toLower(t));
        }
        else if (a.size() > 1 && a[0] == '-') { std::cerr << "Unknown option: " << a << "\n"; return 2; }
        else opt.paths.push_back(a);
    }

    auto rules = buildRules();

    if (opt.help || (opt.paths.empty() && !opt.listRules)) { usage(argv[0]); return opt.help ? 0 : 2; }
    if (opt.listRules) {
        for (const auto& r : rules)
            std::cout << std::left << std::setw(16) << r.id << std::setw(15) << r.category << r.description
                      << (r.needCtx ? "  [context required]" : "") << "\n";
        return 0;
    }

    vector<Finding> findings;
    size_t filesScanned = 0, skipped = 0;

    auto scanFile = [&](const fs::path& p) {
        std::error_code ec;
        auto sz = fs::file_size(p, ec);
        if (ec || sz > opt.maxBytes || looksBinary(p)) { ++skipped; return; }
        std::ifstream in(p);
        if (!in) { ++skipped; return; }
        scanStream(in, p.string(), rules, opt, findings);
        ++filesScanned;
    };

    for (const auto& path : opt.paths) {
        if (path == "-") { scanStream(std::cin, "<stdin>", rules, opt, findings); ++filesScanned; continue; }
        std::error_code ec;
        if (fs::is_directory(path, ec)) {
            for (fs::recursive_directory_iterator it(path, fs::directory_options::skip_permission_denied, ec), end;
                 it != end; it.increment(ec)) {
                if (ec) break;
                const auto& name = it->path().filename().string();
                if (it->is_directory(ec) && (name == ".git" || name == "node_modules")) { it.disable_recursion_pending(); continue; }
                if (it->is_regular_file(ec)) scanFile(it->path());
            }
        } else if (fs::is_regular_file(path, ec)) {
            scanFile(path);
        } else {
            std::cerr << "Cannot access: " << path << "\n";
            ++skipped;
        }
    }

    printFindings(findings, opt);
    printSummary(findings, filesScanned, skipped, (opt.json || opt.csv) ? std::cerr : std::cout);
    return findings.empty() ? 0 : 1;
}
