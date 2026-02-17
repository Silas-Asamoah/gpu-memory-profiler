# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in GPU Memory Profiler, please follow these steps:

### 1. **DO NOT** create a public GitHub issue

Security vulnerabilities should be reported privately to prevent potential exploitation.

### 2. Report the vulnerability

Please report security vulnerabilities to our security team:

**Email**: prince.agyei.tuffour@gmail.com

**Subject**: `[SECURITY] GPU Memory Profiler - [Brief Description]`

### 3. Include the following information

When reporting a vulnerability, please include:

- **Description**: A clear description of the vulnerability
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Impact**: Potential impact of the vulnerability
- **Environment**: OS, Python version, PyTorch/TensorFlow version, CUDA version
- **Proof of concept**: If possible, include a minimal code example
- **Suggested fix**: If you have ideas for fixing the vulnerability

### 4. What happens next

1. **Acknowledgment**: You will receive an acknowledgment within 48 hours
2. **Investigation**: Our security team will investigate the report
3. **Fix development**: If confirmed, we will develop a fix
4. **Release**: A security patch will be released as soon as possible
5. **Credit**: You will be credited in the security advisory (if desired)

### 5. Disclosure timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 7 days
- **Fix release**: As soon as possible, typically within 30 days
- **Public disclosure**: After the fix is released

## Security Best Practices

### For Users

1. **Keep updated**: Always use the latest version of GPU Memory Profiler
2. **Monitor dependencies**: Regularly update your dependencies
3. **Review code**: Review the code you're running, especially in production
4. **Use virtual environments**: Isolate your Python environments
5. **Limit permissions**: Run with minimal required permissions

### For Contributors

1. **Follow secure coding practices**: Validate inputs, handle errors properly
2. **Review security implications**: Consider security impact of changes
3. **Test thoroughly**: Ensure your code doesn't introduce vulnerabilities
4. **Update dependencies**: Keep development dependencies updated
5. **Report issues**: If you find security issues, report them immediately

## Security Features

GPU Memory Profiler includes several security features:

- **Input validation**: All inputs are validated to prevent injection attacks
- **Error handling**: Secure error handling prevents information disclosure
- **Dependency scanning**: Regular dependency vulnerability scanning
- **Code review**: All code changes undergo security review
- **Testing**: Comprehensive security testing in CI/CD pipeline

## Responsible Disclosure

We follow responsible disclosure practices:

- **Private reporting**: Vulnerabilities are reported privately first
- **Coordinated release**: Fixes are released before public disclosure
- **Credit given**: Researchers are credited for their findings
- **No retaliation**: We welcome security research and won't retaliate

## Security Contacts

### Primary Security Contact

- **Email**: prince.agyei.tuffour@gmail.com
- **PGP Key**: Not currently available

### Backup Contacts

- **Prince Agyei Tuffour**: [GitHub](https://github.com/nanaagyei)
- **Silas Bempong**: [GitHub](https://github.com/Silas-Asamoah)

## Security Hall of Fame

We would like to thank the following security researchers for their contributions:

- [To be added as vulnerabilities are reported and fixed]

---

**Last Updated:** February 2026
**Version:** 1.0
