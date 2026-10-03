set -ex

cd frontend
pnpm install
STATIC_EXPORT=1 pnpm next build
rm -rf ../hibiki_zero/static
mkdir -p ../hibiki_zero/static
cp -r out/* ../hibiki_zero/static/