# Set up credential injection

Credential injection lets an agent call an API (GitHub, OpenAI, your own service) without the
secret ever entering the sandbox. The secret stays in **AWS Secrets Manager**. SPACE's network
gateway intercepts the sandbox's TLS traffic and writes the secret into one HTTP header, only
for the hosts you choose. Intercepting TLS needs a certificate authority, which is **AWS
Private CA**.

> **Status: experimental.** This guide follows SPACE's source and its own AWS deployment
> (`infra/space/aws/eks` at SPACE commit `b6ca1b2`), but this project has not yet run it on a
> real AWS account. Step 6 checks your setup end to end and names the broken part. If you run
> it, please open an issue with the result.

## Who does what

| Step | Who | How often |
|---|---|---|
| 1. Check the prerequisites | SPACE operator | once |
| 2. Create the Private CA and give its root to SPACE | SPACE operator | once |
| 3. Create the Secrets Manager reader role | SPACE operator | once |
| 4. Configure SPACE | SPACE operator | once |
| 5. Store each secret, tagged with your organization | app developer | per secret |
| 6. Check the setup end to end | either | after any change |
| 7. Use it from langchain-space | app developer | in code |

If you deploy SPACE with its Terraform module (`infra/space/aws/eks/modules/space`), steps 3
and most of 4 are done for you: see the boxes marked **Terraform**.

## 1. Prerequisites

- SPACE deployed **on AWS**: an AWS account, credentials for SPACE's credential-manager
  authority (`space-credmgr-authority`), and the AWS CLI for the commands below.
- **Not a local Kind install.** As shipped, Kind cannot do this: its charts have no setting
  for the Private CA, its seed build does not take the CA's root certificate, and its default
  storage mode (`SPACE_KIND_STORAGE_CREDENTIALS=exchange`) turns the Secrets Manager
  integration off (SPACE `infra/space/kind/README.md`).
- **Cost.** AWS Private CA is billed per CA per month, a few hundred US dollars for one CA in
  general-purpose mode at the time of writing, plus a small fee per certificate. Secrets
  Manager costs cents per secret per month. Check AWS's current pricing before step 2. One
  CA serves the whole SPACE deployment.

Set these once in your shell for the commands below:

```bash
export AWS_REGION=eu-west-1            # the region SPACE runs in
export ACCOUNT_ID=123456789012
export AUTHORITY_ROLE_ARN=arn:aws:iam::$ACCOUNT_ID:role/space-credmgr-authority  # the AWS
# identity the credential-manager authority runs as (on EKS, its Pod Identity role)
```

## 2. Create the Private CA

SPACE signs with `SHA256WITHECDSA`, so the CA needs an **EC** key. SPACE then issues one
intermediate per node under it, and one more level below for each gateway, so use a root
with no path-length limit (the `RootCACertificate/V1` template below has none).

```bash
CA_ARN=$(aws acm-pca create-certificate-authority \
  --certificate-authority-type ROOT \
  --certificate-authority-configuration '{
      "KeyAlgorithm": "EC_prime256v1",
      "SigningAlgorithm": "SHA256WITHECDSA",
      "Subject": {"CommonName": "SPACE TLS interception root", "Organization": "Your company"}}' \
  --query CertificateAuthorityArn --output text)
aws acm-pca wait certificate-authority-csr-created --certificate-authority-arn "$CA_ARN"

# Self-sign the root and install it on the CA.
aws acm-pca get-certificate-authority-csr --certificate-authority-arn "$CA_ARN" \
  --output text > root.csr
CERT_ARN=$(aws acm-pca issue-certificate --certificate-authority-arn "$CA_ARN" \
  --csr fileb://root.csr --signing-algorithm SHA256WITHECDSA \
  --template-arn arn:aws:acm-pca:::template/RootCACertificate/V1 \
  --validity Value=10,Type=YEARS --query CertificateArn --output text)
aws acm-pca wait certificate-issued --certificate-authority-arn "$CA_ARN" --certificate-arn "$CERT_ARN"
aws acm-pca get-certificate --certificate-authority-arn "$CA_ARN" --certificate-arn "$CERT_ARN" \
  --query Certificate --output text > root-ca.pem
aws acm-pca import-certificate-authority-certificate --certificate-authority-arn "$CA_ARN" \
  --certificate fileb://root-ca.pem
echo "$CA_ARN"
```

Let the authority sign under it:

```bash
aws iam put-role-policy --role-name "${AUTHORITY_ROLE_ARN##*/}" \
  --policy-name space-credmgr-private-ca --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Action\": [\"acm-pca:IssueCertificate\", \"acm-pca:GetCertificate\"],
      \"Resource\": \"$CA_ARN\"}]}"
```

**Sandboxes must trust the root.** SPACE bakes `root-ca.pem` into the **seed** template when
it builds it: build-manager reads it from `/space/etc/space-mitm/root-ca.pem`. Every template
built from seed (such as `code-runtime`) inherits it. After adding the root, **rebuild seed,
then every template built from it**. Sandboxes from older templates will fail TLS.

> **Terraform:** set `credmgr_intermediate_ca_arn = <CA_ARN>`,
> `mitm_root_ca_pem = file("root-ca.pem")` and `mitm_root_ca_external_secret_enabled = true`.
> The module grants the CA permissions and mounts the root into build-manager.

## 3. Create the Secrets Manager reader role

The authority never reads a secret with its own identity. For each read it assumes this role,
tagged with the sandbox's organization, and the role can only read secrets carrying the same
tag. One organization can therefore never read another's secrets. These policies are SPACE's
(`infra/space/aws/eks/modules/space/credmgr_secrets_reader.tf`), written out as JSON.

> **Terraform:** the module creates this role; its ARN is the output
> `credmgr_secrets_reader_role_arn`. Skip to step 4.

```bash
cat > reader-trust.json <<EOF
{"Version": "2012-10-17", "Statement": [{
  "Sid": "OnlyCredmgrAuthorityMayAssumePerOrganization",
  "Effect": "Allow",
  "Principal": {"AWS": "$AUTHORITY_ROLE_ARN"},
  "Action": ["sts:AssumeRole", "sts:TagSession"],
  "Condition": {"Null": {"aws:RequestTag/space-org-id": "false"}}}]}
EOF

cat > reader-policy.json <<'EOF'
{"Version": "2012-10-17", "Statement": [
  {"Sid": "ReadOnlySecretsTaggedForTheSessionOrganization", "Effect": "Allow",
   "Action": "secretsmanager:GetSecretValue", "Resource": "arn:aws:secretsmanager:*:*:secret:*",
   "Condition": {"StringEquals": {
     "secretsmanager:ResourceTag/space-org-id": "${aws:PrincipalTag/space-org-id}"}}},
  {"Sid": "DecryptOnlyThroughSecretsManager", "Effect": "Allow",
   "Action": "kms:Decrypt", "Resource": "*",
   "Condition": {"StringLike": {"kms:ViaService": "secretsmanager.*.amazonaws.com"}}},
  {"Sid": "DenyKmsOutsideSecretsManager", "Effect": "Deny",
   "Action": "kms:*", "Resource": "*",
   "Condition": {"StringNotLike": {"kms:ViaService": "secretsmanager.*.amazonaws.com"}}},
  {"Sid": "DenySecretsNotTaggedForTheSessionOrganization", "Effect": "Deny",
   "Action": "secretsmanager:*", "Resource": "*",
   "Condition": {"StringNotEquals": {
     "secretsmanager:ResourceTag/space-org-id": "${aws:PrincipalTag/space-org-id}"}}},
  {"Sid": "DenySessionsWithoutAnOrganization", "Effect": "Deny",
   "Action": "secretsmanager:*", "Resource": "*",
   "Condition": {"Null": {"aws:PrincipalTag/space-org-id": "true"}}}]}
EOF

READER_ROLE_ARN=$(aws iam create-role --role-name space-credmgr-secrets-reader \
  --assume-role-policy-document file://reader-trust.json --query Role.Arn --output text)
aws iam put-role-policy --role-name space-credmgr-secrets-reader \
  --policy-name space-credmgr-secrets-reader --policy-document file://reader-policy.json

# The authority may assume it, with the organization tag.
aws iam put-role-policy --role-name "${AUTHORITY_ROLE_ARN##*/}" \
  --policy-name space-credmgr-assume-secrets-reader --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{\"Effect\": \"Allow\", \"Action\": [\"sts:AssumeRole\", \"sts:TagSession\"],
                     \"Resource\": \"$READER_ROLE_ARN\"}]}"
echo "$READER_ROLE_ARN"
```

`reader-policy.json` is quoted (`'EOF'`) on purpose: `${aws:PrincipalTag/space-org-id}` is an
IAM variable that AWS fills in, not a shell variable.

Secrets in **another AWS account** also need a resource policy allowing the reader role, and
must use a customer-managed KMS key whose policy allows the role `kms:Decrypt` with
`kms:ViaService = secretsmanager.<region>.amazonaws.com`.

## 4. Configure SPACE

| Service | Setting | Value |
|---|---|---|
| `space-credmgr-authority` | `SPACE_CREDMGR_INTERMEDIATE_CA_ARN` | `$CA_ARN` (step 2) |
| `space-credmgr-authority` | `SPACE_CREDMGR_SECRETS_MANAGER_ROLE_ARN` | `$READER_ROLE_ARN` (step 3), or the Terraform output |
| `space-credmgr-authority` | `AWS_REGION` and AWS credentials | the identity `$AUTHORITY_ROLE_ARN` |
| sandbox-manager | `SPACE_SBXMGR_NETWORK_CREDENTIALS_ENABLED` | `true` (Helm value `networkCredentials.enabled`) |

All of these default to off: with any one missing, requests leave the sandbox **without** the
secret and nothing reports an error, which is why step 6 exists. Restart the services after
changing them. A malformed role ARN stops the authority at startup.

## 5. Store each secret

Each secret must be tagged with **the SPACE organization that will use it**:
`space-org-id=<organization ID>`. Ask your SPACE operator for the ID; a fresh local install's
default organization is `00000000-0000-0000-0000-000000000001`.

The secret is used **verbatim as the header value**. If the API expects `Authorization: Bearer
<token>`, store `Bearer <token>`, not just the token.

```bash
printf 'Bearer %s' "$GITHUB_TOKEN" > token.txt   # keeps the token out of shell history
aws secretsmanager create-secret --name agents/github-token \
  --secret-string file://token.txt \
  --tags Key=space-org-id,Value=<organization ID> \
  --query ARN --output text
rm token.txt
```

Use the full ARN it prints, including the random suffix (`...:secret:agents/github-token-AbCdEf`).

## 6. Check the setup end to end

Create a **throwaway** secret: the check sends the injected header to a public echo service,
which returns it to the sandbox. Never use a real secret here.

```bash
CHECK_VALUE="Bearer lcs-check-$(openssl rand -hex 8)"
CHECK_ARN=$(aws secretsmanager create-secret --name lcs-check-$RANDOM \
  --secret-string "$CHECK_VALUE" --tags Key=space-org-id,Value=<organization ID> \
  --query ARN --output text)

uv run python examples/check_credential_injection.py \
  --secret-arn "$CHECK_ARN" --expect "$CHECK_VALUE"     # SPACE_API_URL etc. must be set

aws secretsmanager delete-secret --secret-id "$CHECK_ARN" --force-delete-without-recovery
```

It creates a sandbox with an injection rule, sends `Authorization: lcs-placeholder` from
inside it, and reads what the echo service received. It always deletes the sandbox and
retires the reference. Use `--echo-url` for another echo service that returns the request's
headers as JSON (`postman-echo.com/headers`, the default, and `httpbin.org/headers` both do).

| The check says | Broken part | Fix |
|---|---|---|
| request never reached the echo service | egress policy or DNS | allow the host and the public resolvers; `build_network_policy` does both |
| TLS failed inside the sandbox | trust in the interception CA | step 2: CA ARN set on the authority, seed and its templates rebuilt with `root-ca.pem` |
| request went out **without** the secret | injection is off, or the secret cannot be read | step 4 settings; step 5 tag matches your organization; step 3 policies; same region |
| injected a value, but not the expected one | the stored value | store the exact header value, including `Bearer ` |
| OK | nothing | |

## 7. Use it from langchain-space

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain_space import (
    CredentialRule,
    HttpSpaceClient,
    SecretGuard,
    SpaceCheckpointer,
    build_network_policy,
    register,
    retire,
)

client = HttpSpaceClient.from_env()
github = register(
    client,
    CredentialRule(
        host="api.github.com",
        header="authorization",
        secret_arn="arn:aws:secretsmanager:eu-west-1:123456789012:secret:agents/github-token-AbCdEf",
    ),
)
policy = build_network_policy([github], allow_domains=["pypi.org"])  # everything else is blocked

saver = SpaceCheckpointer(
    InMemorySaver(),
    client,
    template="code-runtime",
    network_policy=policy,
    # Refuse to send the real token into the sandbox if it also sits in this process's env.
    backend_kwargs={"content_check": SecretGuard.from_env("GITHUB_TOKEN")},
)
```

Inside the sandbox, the agent sends any placeholder (`curl -H 'Authorization: x' ...`) and
the gateway replaces it. `CredentialRule` rejects, before contacting SPACE, what SPACE would
refuse: wildcard hosts, reserved headers, methods other than GET, POST and HEAD. When a
credential is no longer needed, apply a policy without it, then `retire(client, github)`. The
secret itself stays in Secrets Manager.
