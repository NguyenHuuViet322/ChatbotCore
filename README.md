# 📚 Company Chatbot API

Chatbot nội bộ sử dụng tài liệu công ty và tìm kiếm web, xây dựng với FastAPI, LangChain, Gemini (Google Generative AI) và Tavily.

---

## 🚀 Tính năng

- Chat với bot về tài liệu nội bộ hoặc thông tin ngoài internet.
- Tìm kiếm thông minh trong file PDF, DOCX, TXT nhờ mô hình embedding đa ngôn ngữ.
- Sử dụng LLM Gemini 2.0 Flash (Google Generative AI).
- Tìm kiếm web với Tavily Search.
- Cung cấp API chuẩn RESTful qua FastAPI.

---

## 🧩 Kiến trúc hệ thống

```
Client (React/Next.js)
      │
      ▼
FastAPI endpoint (/chat)
      │
      ▼
LangChain Agent
 ├── Tool: Tìm kiếm tài liệu nội bộ
 └── Tool: Tìm kiếm web (Tavily)
      │
      ▼
 Gemini LLM trả lời người dùng
```

---

## 📁 Cấu trúc thư mục

```
.
├── app.py                    # FastAPI app chính
├── vector_store.py          # Xử lý tài liệu và vectorstore
├── .env                     # API keys (Google, Tavily)
├── data/                    # Chứa tài liệu nội bộ (PDF, DOCX, TXT...)
├── vectorstore/             # Dữ liệu đã mã hóa được lưu tại đây
├── requirements.txt         # Danh sách thư viện cần cài
└── README.md                # Hướng dẫn sử dụng
```

---

## 🧪 Yêu cầu hệ thống

- Python >= 3.10
- Có thể sử dụng GPU để tăng tốc, hoặc CPU cũng được.
- Tài khoản API:
  - [Google Generative AI](https://makersuite.google.com/)
  - [Tavily Search](https://www.tavily.com/)

---

## ⚙️ Cài đặt

### 1. Clone source

```bash
git clone https://github.com/NguyenHuuViet322/ChatbotCore
cd ChatbotCore
```

### 2. Tạo môi trường ảo và cài thư viện

```bash
python -m venv venv
source venv/bin/activate        # Với Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Tạo file `.env`

Tạo file `.env` trong thư mục gốc với nội dung sau:

```
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY=your_google_api_key
```

> Thay `your_tavily_api_key` và `your_google_api_key` bằng khóa thật của bạn.

### 4. Thêm tài liệu

Đặt các file PDF, DOCX, TXT... vào thư mục `./data`

---

## ▶️ Chạy server

```bash
uvicorn app:app --reload
```

Mặc định chạy tại: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🧪 Gọi API

Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Ví dụ request `/chat`

```json
POST /chat

{
  "session_id": "abc123",
  "messages": [
    { "role": "user", "content": "Công ty có chính sách nghỉ phép như thế nào?" }
  ]
}
```

### Phản hồi mẫu

```json
{
  "answer": "Theo tài liệu nội bộ, nhân viên được nghỉ phép 12 ngày mỗi năm..."
}
```

---

## 📌 Ghi chú

- Kết quả phụ thuộc vào chất lượng tài liệu và API key còn hạn hay không.
- Nếu không có GPU, hệ thống vẫn chạy được nhưng tốc độ xử lý chậm hơn.

---

## 📃 Giấy phép

Dự án phát hành theo giấy phép MIT. Bạn có thể sử dụng và sửa đổi theo nhu cầu.
