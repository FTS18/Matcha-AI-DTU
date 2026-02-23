/** @type {import('next').NextConfig} */
const nextConfig = {
    transpilePackages: [
        '@matcha/shared',
        '@react-pdf/renderer',
        'socket.io-client',
        'engine.io-client',
        'engine.io-parser'
    ],
    webpack: (config) => {
        return config;
    },
    async headers() {
        return [
            {
                source: '/(.*)',
                headers: [
                    {
                        key: 'Cross-Origin-Opener-Policy',
                        value: 'same-origin',
                    },
                    {
                        key: 'Cross-Origin-Embedder-Policy',
                        value: 'require-corp',
                    },
                ],
            },
        ];
    },
};

module.exports = nextConfig;
