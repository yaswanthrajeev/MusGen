import { AuthView } from "@daveyplate/better-auth-ui"
import { authViewPaths } from "@daveyplate/better-auth-ui/server"

export const dynamicParams = false

export function generateStaticParams() {
    return Object.values(authViewPaths).map((path) => ({ pathname: path }))
}

export default async function AuthPage({ params }: { params: Promise<{ pathname: string }> }) {
    const { pathname } = await params
    
    return (
        <main className="container flex grow flex-col items-center justify-center self-center p-4 md:p-6">
            <AuthView path={pathname} />
        </main>
    )
}