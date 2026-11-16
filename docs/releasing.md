# Publishing releases

The [Publish to PyPI workflow](../.github/workflows/publish.yml) builds and checks
a source distribution and a wheel. A published GitHub release triggers the PyPI
upload; ordinary pushes, pull requests and manual validation runs do not publish.

The wheel includes the native sources. CPU operators compile on first use, so
users still need the compiler and OpenMP prerequisites in the
[installation guide](source/getting-started/installation.md). This workflow does
not publish the host-specific binaries built by `tools/build_tensor_wheel.py`.

## One-time account setup

Keep your personal account as the only PyPI project owner. GitHub Actions uses
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
to upload releases without a saved API token.

Once the repository is public, open **Settings → Environments** and create an
environment named `pypi`. Add yourself as its required reviewer. Leave
**Prevent self-review** unchecked so you can approve releases you initiate.
You can restrict deployment tags to `v*`.
These environment settings must be configured on GitHub; the workflow file does
not create the reviewer rule. See
[GitHub's environment instructions](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).

In your [PyPI account's Publishing settings](https://pypi.org/manage/account/publishing/),
add a pending GitHub publisher:

| Field | Value |
| --- | --- |
| PyPI project name | `tabgen-eval` |
| Repository owner | `granica-ai` |
| Repository name | `tabgen-eval` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

Use the filename alone, without `.github/workflows/`. The first successful upload
creates the PyPI project under your account. A pending publisher does not reserve
the project name. If the project already exists under your account, add the
publisher in that project's publishing settings instead.

## Validate without publishing

On GitHub, open **Actions → Publish to PyPI → Run workflow** and select `main`.
The build job checks the distributions and saves them as a workflow artifact;
the publishing job is skipped. Review this run and the existing installation CI
before creating a release.

## Publish a version

1. Set the new version in `pyproject.toml`, refresh `uv.lock` with `uv lock`,
   commit both files and push to `main`. The initial version is `0.1.0`;
   subsequent releases need a new version.
2. On GitHub, create a release from `main` with a tag matching `v` followed by
   that exact version, for example `v0.1.0`. Publish the release.
3. Review the **Publish to PyPI** build, then approve the `pypi` environment job.
   This approval is required only when the environment's reviewer rule is
   configured; otherwise publication follows a successful build automatically.
4. After upload, verify the release in a fresh environment on a supported
   platform:

   ```bash
   python -m pip install "tabgen-eval==0.1.0"
   python -c "from importlib.metadata import version; print(version('tabgen-eval'))"
   ```

The workflow rejects a release tag that does not match `pyproject.toml`. PyPI
does not allow replacing an uploaded distribution; use a new version for a
changed release.
