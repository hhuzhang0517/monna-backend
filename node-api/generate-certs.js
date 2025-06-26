const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// 证书输出路径
const SSL_DIR = path.resolve(__dirname, '../');
const KEY_PATH = path.join(SSL_DIR, 'private.key');
const CERT_PATH = path.join(SSL_DIR, 'certificate.crt');
const CSR_PATH = path.join(SSL_DIR, 'csr.pem');

console.log('开始生成SSL自签名证书...');
console.log(`证书将保存到: ${SSL_DIR}`);

try {
  // 确保目录存在
  if (!fs.existsSync(SSL_DIR)) {
    fs.mkdirSync(SSL_DIR, { recursive: true });
    console.log(`创建目录: ${SSL_DIR}`);
  }

  // 生成私钥
  console.log('生成私钥...');
  execSync(`openssl genrsa -out "${KEY_PATH}" 2048`);

  // 生成CSR
  console.log('生成证书签名请求(CSR)...');
  execSync(`openssl req -new -key "${KEY_PATH}" -out "${CSR_PATH}" -subj "/CN=localhost"`);

  // 生成自签名证书
  console.log('生成自签名证书...');
  execSync(`openssl x509 -req -days 365 -in "${CSR_PATH}" -signkey "${KEY_PATH}" -out "${CERT_PATH}"`);

  console.log('SSL证书生成完成!');
  console.log(`私钥路径: ${KEY_PATH}`);
  console.log(`证书路径: ${CERT_PATH}`);
} catch (error) {
  console.error('证书生成失败:', error.message);
  process.exit(1);
} 