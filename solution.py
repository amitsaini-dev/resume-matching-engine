import math

SKILL_ALIASES = {
    "python": "python", "pyhton": "python", "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp", "r": "r", "kotlin": "kotlin",
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning",
    "deep-learning": "deep_learning", "tensorflow": "tensorflow",
    "pytorch": "pytorch", "keras": "keras", "nlp": "nlp", "bert": "bert",
    "xgboost": "xgboost", "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css", "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask", "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    "docker": "docker", "kubernetes": "kubernetes", "kubernates": "kubernetes",
    "k8s": "kubernetes", "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws", "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

RESUMES = [
    ("Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]

JDS = {
    "JD1 - Kakao ML Engineer": [
        "python","machine learning","deep learning","tensorflow","pytorch","sql",
        "data visualization","nlp","bert","feature engineering","statistics"
    ],
    "JD2 - Naver Backend Engineer": [
        "java","spring boot","mysql","postgresql","microservices","docker",
        "kubernetes","rest api","ci/cd","redis"
    ],
    "JD3 - Line Frontend Engineer": [
        "javascript","react","vue","typescript","rest api","html/css",
        "node.js","graphql","redux","jest","aws"
    ],
}

# Sort multi-word aliases by length (longest first) for greedy matching
multi_word = sorted(
    [(k, v) for k, v in SKILL_ALIASES.items() if ' ' in k or '-' in k],
    key=lambda x: -len(x[0])
)

def normalize(raw):
    tokens = [t.strip().lower() for t in raw.split(',')]
    result, seen = [], set()
    for token in tokens:
        matched = False
        for phrase, canonical in multi_word:
            if phrase in token:
                if canonical not in seen:
                    result.append(canonical); seen.add(canonical)
                matched = True; break
        if not matched and token in SKILL_ALIASES:
            canonical = SKILL_ALIASES[token]
            if canonical not in seen:
                result.append(canonical); seen.add(canonical)
    return result

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb) if na and nb else 0.0

# Step 1: Normalize resumes
normalized = [(name, normalize(raw)) for name, raw in RESUMES]

# Step 2: Build vocabulary
vocab = sorted({s for _, skills in normalized for s in skills})

# Step 3: Compute IDF
df = {skill: sum(1 for _, skills in normalized if skill in skills) for skill in vocab}
idf = {skill: math.log(10 / df[skill]) for skill in vocab}

# Step 4: TF-IDF vectors
tfidf = []
for name, skills in normalized:
    N = len(skills)
    vec = [(1/N * idf[s]) if s in skills else 0.0 for s in vocab]
    tfidf.append((name, vec))

# Step 5 & 6: JD binary vectors + cosine similarity
for jd_name, jd_skills in JDS.items():
    jd_canonical = set()
    for s in jd_skills:
        s = s.lower()
        matched = False
        for phrase, canonical in multi_word:
            if phrase == s:
                jd_canonical.add(canonical); matched = True; break
        if not matched and s in SKILL_ALIASES:
            jd_canonical.add(SKILL_ALIASES[s])
    jd_vec = [1.0 if s in jd_canonical else 0.0 for s in vocab]
    scores = sorted([(name, cosine(vec, jd_vec)) for name, vec in tfidf],
                    key=lambda x: (-x[1], x[0]))
    print(f"{jd_name}")
    print("  " + ", ".join(f"{n}({s:.2f})" for n, s in scores[:3]))
