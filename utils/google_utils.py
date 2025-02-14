from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from settings import settings


# https://console.cloud.google.com/marketplace/product/google/drive.googleapis.com
def upload_google_drive(file_name: str):
    mime_types = {
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'xls': 'application/vnd.ms-excel',
        'csv': 'text/csv',
        'json': 'application/json',
        'html': 'text/html',
        'txt': 'text/plain',
        'xml': 'application/xml',
    }

    ext = file_name.split(".")[-1]
    mime_type = mime_types.get(ext)
    if not mime_type:
        print(f'지원하지 않는 파일 확장자입니다: {file_name}')
        return None

    # 구글 서비스 계정 인증
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    service_account_info = settings.GOOGLE_SERVICE_ACCOUNT_DICT
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=creds)

    # 기존 파일 ID 찾기
    query = f"name = '{file_name}' and mimeType = '{mime_type}'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        file_id = files[0]['id']  # 기존 파일 ID 가져오기
        print(f"기존 파일 업데이트: {file_name} (ID: {file_id})")

        # 기존 파일 덮어쓰기 (업데이트)
        media = MediaFileUpload(file_name, mimetype=mime_type, resumable=True)
        updated_file = drive_service.files().update(fileId=file_id, media_body=media).execute()
    else:
        # 기존 파일이 없으면 새로 생성
        file_metadata = {"name": file_name, "mimeType": mime_type}
        media = MediaFileUpload(file_name, mimetype=mime_type)
        new_file = drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        file_id = new_file.get("id")
        print(f"새 파일 생성: {file_name} (ID: {file_id})")

    # 파일을 공개 링크로 설정
    drive_service.permissions().create(
        fileId=file_id,
        body={"role": "reader", "type": "anyone"},
    ).execute()

    # 공유 링크
    if ext == 'html':
        return f"https://drive.google.com/uc?export=view&id={file_id}"
    else:
        return f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"


def list_google_drive():
    # 구글 서비스 계정 인증
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    SERVICE_ACCOUNT_FILE = "../credentials.json"  # 구글 서비스 계정 JSON 키 파일 경로
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=creds)

    # 전체 파일 목록 조회
    results = drive_service.files().list(fields="files(id, name, mimeType)").execute()
    files = results.get("files", [])

    if not files:
        print("업로드된 파일이 없습니다.")
        return []

    # 파일 목록
    file_links = []
    for file in files:
        file_id = file["id"]
        file_name = file["name"]
        file_link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        file_links.append((file_name, file_link))

    return file_links


def delete_google_drive(file_id: str):
    # 구글 서비스 계정 인증
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    SERVICE_ACCOUNT_FILE = "../credentials.json"  # 구글 서비스 계정 JSON 키 파일 경로
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=creds)
    drive_service.files().delete(fileId=file_id).execute()


